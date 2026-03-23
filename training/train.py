"""
training/train.py
-----------------
Contrastive pre-training loop for the fMRI spatiotemporal encoder.

Supports both single-GPU and multi-GPU (DDP) training.

Single-GPU usage
~~~~~~~~~~~~~~~~
    python training/train.py --config configs/default.yaml

Multi-GPU (2 GPUs on 1 node)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    torchrun --nproc_per_node=2 training/train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.optim as optim
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Local imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.encoder       import Encoder
from model.projection_head import ProjectionHead
from training.augmentations  import augment_spatial_pair, augment_temporal
from training.contrastive_loss import nt_xent_loss
from training.dataset        import SubjectICADataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait      = 0
        self.stop      = False

    def __call__(self, loss: float) -> None:
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.wait      = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop = True


def is_main_process() -> bool:
    """True on rank-0 (or when not using DDP)."""
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def setup_ddp() -> tuple[int, torch.device]:
    """Initialise the process group for DDP training."""
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, device


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_one_epoch(
    encoder,
    projection_head,
    loader,
    optimizer,
    device,
    temperature: float,
) -> float:
    encoder.train()
    projection_head.train()
    total_loss = 0.0

    for batch in loader:
        pre_spatial  = batch["pre_spatial"].to(device)
        post_spatial = batch["post_spatial"].to(device)
        pre_temporal = batch["pre_temporal"].to(device)
        post_temporal= batch["post_temporal"].to(device)

        # Apply consistent spatial augmentations to the pair
        pre_spatial_aug, post_spatial_aug = augment_spatial_pair(
            [pre_spatial, post_spatial], device=device
        )

        # Apply independent temporal augmentations
        pre_temporal_aug  = augment_temporal(pre_temporal)
        post_temporal_aug = augment_temporal(post_temporal)

        # Forward pass
        z_pre  = projection_head(encoder(pre_spatial_aug,  pre_temporal_aug))
        z_post = projection_head(encoder(post_spatial_aug, post_temporal_aug))

        loss = nt_xent_loss(z_pre, z_post, temperature=temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict) -> None:
    # ---- distributed setup ----
    use_ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if use_ddp:
        rank, device = setup_ddp()
    else:
        rank   = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data ----
    df      = pd.read_csv(cfg["data"]["csv_path"])
    dataset = SubjectICADataset(
        df,
        temporal_resample=cfg["data"].get("temporal_input_dim"),
    )

    sampler = DistributedSampler(dataset) if use_ddp else None
    loader  = DataLoader(
        dataset,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = (sampler is None),
        sampler     = sampler,
        num_workers = cfg["data"].get("num_workers", 4),
        pin_memory  = True,
    )

    # ---- model ----
    encoder = Encoder(
        temporal_input_dim = cfg["data"].get("temporal_input_dim", 590),
        embedding_dim      = cfg["model"].get("embedding_dim", 128),
    ).to(device)

    projection_head = ProjectionHead(
        input_dim  = cfg["model"].get("embedding_dim", 128),
        hidden_dim = cfg["model"].get("projection_hidden_dim", 64),
        output_dim = cfg["model"].get("projection_dim", 32),
    ).to(device)

    if use_ddp:
        encoder         = DDP(encoder,         device_ids=[device])
        projection_head = DDP(projection_head, device_ids=[device])

    # ---- optimiser ----
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(projection_head.parameters()),
        lr           = cfg["training"].get("lr", 1e-4),
        weight_decay = cfg["training"].get("weight_decay", 0.0),
    )

    early_stopping = EarlyStopping(
        patience  = cfg["training"].get("early_stopping_patience", 20),
        min_delta = cfg["training"].get("early_stopping_min_delta", 0.01),
    )

    # ---- output dir ----
    out_dir = cfg.get("output_dir", "outputs")
    if is_main_process():
        os.makedirs(out_dir, exist_ok=True)

    # ---- training loop ----
    train_losses = []
    best_loss    = float("inf")
    temperature  = cfg["training"].get("temperature", 0.05)
    epochs       = cfg["training"].get("epochs", 150)

    for epoch in range(epochs):
        if use_ddp:
            sampler.set_epoch(epoch)

        epoch_loss = train_one_epoch(
            encoder, projection_head, loader, optimizer, device, temperature
        )
        train_losses.append(epoch_loss)

        if is_main_process():
            print(f"Epoch {epoch+1:>4d}/{epochs}  loss={epoch_loss:.4f}", flush=True)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                _save(encoder,         os.path.join(out_dir, "best_encoder.pth"))
                _save(projection_head, os.path.join(out_dir, "best_projection_head.pth"))

        early_stopping(epoch_loss)
        if early_stopping.stop:
            if is_main_process():
                print("Early stopping triggered.", flush=True)
            break

    # ---- save artefacts ----
    if is_main_process():
        _save(encoder,         os.path.join(out_dir, "final_encoder.pth"))
        _save(projection_head, os.path.join(out_dir, "final_projection_head.pth"))
        np.save(os.path.join(out_dir, "train_losses.npy"), np.array(train_losses))
        print(f"Training complete. Artefacts saved to '{out_dir}'.")

    if use_ddp:
        dist.destroy_process_group()


def _save(model, path: str) -> None:
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(state, path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fMRI contrastive pre-training")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)
