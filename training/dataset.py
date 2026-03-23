"""
training/dataset.py
-------------------
Dataset for subject-specific pre/post surgery ICA pairs.

Each sample returns a matched (pre, post) pair for the same subject and
functional network.  The contrastive loss treats these as positive pairs.

CSV schema required
~~~~~~~~~~~~~~~~~~~
subject_id               e.g.  sub-01
network                  e.g.  Motor | Vision | Language | Frontal | Temporal
condition                pre  or  post
aggregated_spatial_path  path to .nii / .nii.gz  ICA spatial map
aggregated_temporal_path path to .txt  ICA timeseries (one value per row)
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from nilearn.image import load_img
from torch.utils.data import Dataset


class SubjectICADataset(Dataset):
    """Paired pre/post ICA dataset for contrastive pre-training.

    Args:
        df:                  DataFrame with the schema described above.
        temporal_resample:   If set, resample all timeseries to this length.
    """

    def __init__(self, df: pd.DataFrame, temporal_resample: int | None = None):
        self.df               = df
        self.temporal_resample = temporal_resample

        self.subjects = df["subject_id"].unique()
        self.networks = df["network"].unique()

        # subject → network → condition → {spatial_path, temporal_path}
        self.index: dict = defaultdict(lambda: defaultdict(dict))
        for _, row in df.iterrows():
            self.index[row["subject_id"]][row["network"]][row["condition"]] = {
                "spatial_path":  row["aggregated_spatial_path"],
                "temporal_path": row["aggregated_temporal_path"],
            }

        # Build a flat list of valid (subject, network) pairs
        self.pairs = [
            (subj, net)
            for subj in self.subjects
            for net  in self.networks
            if "pre"  in self.index[subj].get(net, {})
            and "post" in self.index[subj].get(net, {})
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        subject, network = self.pairs[idx]
        pre  = self.index[subject][network]["pre"]
        post = self.index[subject][network]["post"]

        return {
            "pre_spatial":   self._load_spatial(pre["spatial_path"]),
            "post_spatial":  self._load_spatial(post["spatial_path"]),
            "pre_temporal":  self._load_temporal(pre["temporal_path"]),
            "post_temporal": self._load_temporal(post["temporal_path"]),
            "subject":       subject,
            "network":       network,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_spatial(self, path: str) -> torch.Tensor:
        """Load a NIfTI ICA map → (1, D, H, W) float32 tensor."""
        vol = load_img(path).get_fdata()                  # (H, W, D)
        vol = np.transpose(vol, (2, 0, 1))                # (D, H, W)
        vol = np.expand_dims(vol, axis=0)                 # (1, D, H, W)
        return torch.tensor(vol, dtype=torch.float32)

    def _load_temporal(self, path: str) -> torch.Tensor:
        """Load a text timeseries → (1, T) float32 tensor."""
        ts = np.loadtxt(path)
        if self.temporal_resample and len(ts) != self.temporal_resample:
            from scipy.signal import resample as sp_resample
            ts = sp_resample(ts, self.temporal_resample)
        ts = np.expand_dims(ts, axis=0)                   # (1, T)
        return torch.tensor(ts, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Downstream classification dataset (single-condition, with labels)
# ---------------------------------------------------------------------------

class ICADataset(Dataset):
    """Single-scan ICA dataset for supervised fine-tuning.

    Label encoding:  0 = pre-surgery,  1 = post-surgery.

    Args:
        csv_file:            Path to CSV (see schema above).
        temporal_resample:   Resample timeseries to this length (optional).
        networks:            Subset of networks to include (None = all).
        transform:           Optional torchvision transform applied to spatial data.
    """

    NETWORKS_DEFAULT = ("Motor", "Vision", "Language", "Frontal", "Temporal")

    def __init__(
        self,
        csv_file: str,
        temporal_resample: int | None = 590,
        networks: tuple[str, ...] | None = None,
        transform=None,
    ):
        df = pd.read_csv(csv_file)
        if networks:
            df = df[df["network"].isin(networks)]
        self.df               = df.reset_index(drop=True)
        self.temporal_resample = temporal_resample
        self.transform        = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        spatial  = self._load_spatial(row["aggregated_spatial_path"])
        temporal = self._load_temporal(row["aggregated_temporal_path"])
        label    = torch.tensor(0 if row["condition"] == "pre" else 1, dtype=torch.long)

        if self.transform:
            spatial = self.transform(spatial)

        return {
            "spatial_data":  spatial,
            "temporal_data": temporal,
            "label":         label,
            "network":       row["network"],
            "subject_id":    row["subject_id"],
        }

    def _load_spatial(self, path: str) -> torch.Tensor:
        vol = load_img(path).get_fdata()
        vol = np.transpose(vol, (2, 0, 1))
        vol = np.expand_dims(vol, axis=0)
        return torch.tensor(vol, dtype=torch.float32)

    def _load_temporal(self, path: str) -> torch.Tensor:
        ts = np.loadtxt(path)
        if self.temporal_resample and len(ts) != self.temporal_resample:
            from scipy.signal import resample as sp_resample
            ts = sp_resample(ts, self.temporal_resample)
        ts = np.expand_dims(ts, axis=0)
        return torch.tensor(ts, dtype=torch.float32)
