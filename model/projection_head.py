"""
model/projection_head.py
------------------------
MLP projection head used during contrastive pre-training only.
Maps encoder embeddings to a lower-dimensional L2-normalised space
where the NT-Xent loss is computed.

The projection head is discarded after pre-training; downstream tasks
use the encoder output directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head with L2 normalisation.

    Architecture:  input_dim → hidden_dim (ReLU) → output_dim (L2-norm)

    Args:
        input_dim:  Dimension of encoder output (default 128).
        hidden_dim: Hidden layer size (default 64).
        output_dim: Projected space dimension used for loss (default 32).
    """

    def __init__(
        self,
        input_dim:  int = 128,
        hidden_dim: int = 64,
        output_dim: int = 32,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) encoder embedding.

        Returns:
            z: (B, output_dim) L2-normalised projection.
        """
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)
