"""
model/encoder.py
----------------
Dual-branch spatiotemporal encoder for resting-state fMRI ICA components.

Spatial branch:  3D CNN operating on volumetric ICA spatial maps (NIfTI).
Temporal branch: 1D CNN operating on ICA timeseries.
Both branches are fused via concatenation and a shared FC layer.
"""

import torch
import torch.nn as nn


class SpatialBranch(nn.Module):
    """3D convolutional encoder for volumetric ICA spatial maps.

    Input shape:  (B, 1, D, H, W)  — single-channel 3D volume
    Output shape: (B, 256)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1,   32,  kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32,  64,  kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv3d(64,  64,  kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(64,  128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Assumes input spatial size (48, 55, 55) → after 3× pool → (6, 6, 5)
        # Adjust flat_dim if your native voxel grid differs.
        self.flat_dim = 128 * 6 * 6 * 5
        self.fc = nn.Linear(self.flat_dim, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(torch.relu(self.conv2(torch.relu(self.conv1(x)))))
        x = self.pool2(torch.relu(self.conv4(torch.relu(self.conv3(x)))))
        x = self.pool3(torch.relu(self.conv6(torch.relu(self.conv5(x)))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return x


class TemporalBranch(nn.Module):
    """1D convolutional encoder for ICA timeseries.

    Input shape:  (B, 1, T)  — single-channel timeseries
    Output shape: (B, 128)
    """

    def __init__(self, temporal_input_dim: int = 590):
        super().__init__()
        self.conv1 = nn.Conv1d(1,  32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool  = nn.MaxPool1d(kernel_size=2, stride=2)

        # Two pooling steps → T // 4
        flat_dim = 64 * (temporal_input_dim // 4)
        self.fc = nn.Linear(flat_dim, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return x


class Encoder(nn.Module):
    """Dual-branch spatiotemporal encoder.

    Combines a 3D spatial CNN and a 1D temporal CNN into a shared 128-d embedding.

    Args:
        temporal_input_dim: Number of fMRI timepoints (default 590).
        embedding_dim:      Dimension of the fused output embedding (default 128).
    """

    def __init__(self, temporal_input_dim: int = 590, embedding_dim: int = 128):
        super().__init__()
        self.spatial_branch  = SpatialBranch()
        self.temporal_branch = TemporalBranch(temporal_input_dim)
        self.fc_combined = nn.Linear(256 + 128, embedding_dim)

    def forward(
        self,
        x_spatial:  torch.Tensor,
        x_temporal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_spatial:  (B, 1, D, H, W) volumetric ICA spatial map.
            x_temporal: (B, 1, T) ICA timeseries.

        Returns:
            embedding: (B, embedding_dim)
        """
        z_s = self.spatial_branch(x_spatial)
        z_t = self.temporal_branch(x_temporal)
        z   = torch.cat([z_s, z_t], dim=1)
        return self.fc_combined(z)
