"""
downstream/classifier.py
------------------------
Fine-tuning a lightweight classifier on top of the frozen pre-trained encoder
for binary surgical outcome prediction (pre vs post surgery).

The encoder weights are loaded and frozen.  Only the small MLP classifier is
trained, using 5-fold subject-stratified cross-validation.

Usage
~~~~~
    python downstream/classifier.py \
        --encoder_weights outputs/best_encoder.pth \
        --csv data/labelled_data.csv \
        --network Motor \
        --n_folds 5
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model.encoder   import Encoder
from training.dataset import ICADataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Classifier head
# ---------------------------------------------------------------------------

class Classifier(nn.Module):
    """Lightweight MLP classifier on top of frozen encoder embeddings.

    Architecture: BN → FC(128, 64) → BN → ReLU → Dropout → FC(64, 2)
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class SubjectAwareEarlyStopping:
    """Stop when neither loss nor accuracy improve."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = float("inf")
        self.best_acc   = 0.0
        self.early_stop = False

    def __call__(self, val_loss: float, val_acc: float) -> None:
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter   = 0
        elif val_acc > (self.best_acc + self.min_delta):
            self.best_acc = val_acc
            self.counter  = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def custom_collate_fn(batch):
    spatial_data  = pad_sequence([b["spatial_data"]  for b in batch], batch_first=True)
    temporal_data = pad_sequence([b["temporal_data"] for b in batch], batch_first=True)
    labels        = torch.stack([b["label"]          for b in batch])
    networks      = [b["network"]    for b in batch]
    subject_ids   = [b["subject_id"] for b in batch]
    return {
        "spatial_data":  spatial_data.to(device),
        "temporal_data": temporal_data.to(device),
        "label":         labels.to(device),
        "network":       networks,
        "subject_id":    subject_ids,
    }


def make_loaders(df, train_subjects, val_subjects, batch_size: int = 32):
    train_df = df[df["subject_id"].isin(train_subjects)].reset_index(drop=True)
    val_df   = df[df["subject_id"].isin(val_subjects)].reset_index(drop=True)

    def _ds(frame):
        return ICADataset.__new__(ICADataset).__dict__.update(
            {"df": frame, "temporal_resample": 590, "transform": None}
        ) or _SimpleDataset(frame)

    train_loader = DataLoader(
        _SimpleDataset(train_df), batch_size=max(2, min(batch_size, len(train_df))),
        shuffle=True,  collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        _SimpleDataset(val_df),   batch_size=1,
        shuffle=False, collate_fn=custom_collate_fn
    )
    return train_loader, val_loader


class _SimpleDataset(ICADataset):
    """Thin wrapper to pass a pre-filtered DataFrame directly."""
    def __init__(self, df: pd.DataFrame):
        self.df               = df
        self.temporal_resample = 590
        self.transform        = None


# ---------------------------------------------------------------------------
# Train / validate / test
# ---------------------------------------------------------------------------

def train_epoch(encoder, clf, loader, criterion, optimizer):
    clf.train()
    running_loss, preds_all, labels_all = 0.0, [], []
    for batch in loader:
        with torch.no_grad():
            emb = encoder(batch["spatial_data"], batch["temporal_data"])
        out  = clf(emb)
        loss = criterion(out, batch["label"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds_all.extend(out.argmax(1).cpu().numpy())
        labels_all.extend(batch["label"].cpu().numpy())
    return running_loss / len(loader), accuracy_score(labels_all, preds_all)


@torch.no_grad()
def evaluate(encoder, clf, loader, criterion):
    clf.eval()
    loss_sum, preds_all, labels_all, probs_all = 0.0, [], [], []
    for batch in loader:
        emb   = encoder(batch["spatial_data"], batch["temporal_data"])
        out   = clf(emb)
        loss_sum += criterion(out, batch["label"]).item()
        preds_all.extend(out.argmax(1).cpu().numpy())
        probs_all.extend(torch.softmax(out, 1)[:, 1].cpu().numpy())
        labels_all.extend(batch["label"].cpu().numpy())
    val_loss = loss_sum / len(loader)
    val_acc  = accuracy_score(labels_all, preds_all)
    return val_loss, val_acc, preds_all, labels_all, probs_all


def full_metrics(labels, preds, probs):
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    return {
        "accuracy":    accuracy_score(labels, preds),
        "precision":   precision_score(labels, preds, zero_division=0),
        "recall":      recall_score(labels, preds, zero_division=0),
        "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1":          f1_score(labels, preds, zero_division=0),
        "auc":         roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(labels, preds),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_learning_curves(history, network, fold, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, title in zip(axes, ["loss", "acc"], ["Loss", "Accuracy"]):
        ax.plot(history[f"train_{key}"], label="Train")
        ax.plot(history[f"val_{key}"],   label="Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.suptitle(f"{network} – Fold {fold+1}")
    fig.savefig(os.path.join(out_dir, f"{network}_curve_fold{fold+1}.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-validation runner
# ---------------------------------------------------------------------------

def run_cv(
    encoder,
    df: pd.DataFrame,
    network: str,
    n_splits:  int = 5,
    epochs:    int = 100,
    out_dir:   str = "outputs",
):
    net_df   = df[df["network"] == network].reset_index(drop=True)
    subjects = net_df["subject_id"].unique()
    s_labels = [net_df[net_df["subject_id"] == s]["label"].iloc[0] for s in subjects]

    skf        = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_stats = {k: [] for k in ["accuracy", "precision", "recall", "f1", "auc",
                                   "sensitivity", "specificity"]}
    criterion  = nn.CrossEntropyLoss()

    for fold, (tr_idx, va_idx) in enumerate(skf.split(subjects, s_labels)):
        print(f"\n--- {network}  Fold {fold+1}/{n_splits} ---")
        tr_subj = subjects[tr_idx]
        va_subj = subjects[va_idx]

        train_loader, val_loader = make_loaders(net_df, tr_subj, va_subj)
        clf       = Classifier().to(device)
        optimizer = torch.optim.Adam(clf.parameters(), lr=5e-4, weight_decay=1e-5)
        es        = SubjectAwareEarlyStopping(patience=10)
        history   = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_state, best_acc = None, 0.0

        for epoch in range(epochs):
            tl, ta = train_epoch(encoder, clf, train_loader, criterion, optimizer)
            vl, va, vp, vlab, vprob = evaluate(encoder, clf, val_loader, criterion)
            history["train_loss"].append(tl)
            history["train_acc"].append(ta)
            history["val_loss"].append(vl)
            history["val_acc"].append(va)

            es(vl, va)
            if va > best_acc:
                best_acc   = va
                best_state = clf.state_dict()

            if (epoch + 1) % 20 == 0:
                print(f"  ep {epoch+1:>3d}  tl={tl:.4f} ta={ta:.4f}  vl={vl:.4f} va={va:.4f}")
            if es.early_stop:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        clf.load_state_dict(best_state)
        _, _, final_preds, final_labels, final_probs = evaluate(
            encoder, clf, val_loader, criterion
        )
        m = full_metrics(final_labels, final_preds, final_probs)

        for k in fold_stats:
            fold_stats[k].append(m[k])

        print(f"  Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}")
        print(f"  Sens={m['sensitivity']:.4f}  Spec={m['specificity']:.4f}")
        print(f"  CM:\n{m['confusion_matrix']}")

        plot_learning_curves(history, network, fold, out_dir)

        model_path = os.path.join(out_dir, f"{network}_clf_fold{fold+1}.pth")
        torch.save(clf.state_dict(), model_path)

    print(f"\n=== {network}  {n_splits}-fold Summary ===")
    for k, vals in fold_stats.items():
        print(f"  {k:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    return fold_stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_weights", required=True)
    parser.add_argument("--csv",             required=True)
    parser.add_argument("--network",         default="Motor")
    parser.add_argument("--n_folds",         type=int, default=5)
    parser.add_argument("--epochs",          type=int, default=100)
    parser.add_argument("--output_dir",      default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load(args.encoder_weights, map_location=device))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"Encoder loaded from {args.encoder_weights}")

    df = pd.read_csv(args.csv)
    df["label"] = (df["condition"] == "post").astype(int)

    run_cv(
        encoder    = encoder,
        df         = df,
        network    = args.network,
        n_splits   = args.n_folds,
        epochs     = args.epochs,
        out_dir    = args.output_dir,
    )
