"""
training/contrastive_loss.py
----------------------------
NT-Xent (Normalized Temperature-scaled Cross Entropy) loss, the SimCLR objective.

In our setting positive pairs are (pre-surgery, post-surgery) embeddings of the
same subject/network.  All other within-batch pairs are treated as negatives.

Reference:
    Chen et al., "A Simple Framework for Contrastive Learning of Visual
    Representations", ICML 2020.  https://arxiv.org/abs/2002.05709
"""

import torch
import torch.nn.functional as F


def nt_xent_loss(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """NT-Xent loss for a batch of positive pairs.

    Assumes embeddings are already L2-normalised (e.g. output of ProjectionHead).
    If not, pass ``normalise=True``.

    Args:
        z_i:         (N, D) embeddings for the first view  (e.g. pre-surgery).
        z_j:         (N, D) embeddings for the second view (e.g. post-surgery).
        temperature: Scaling factor τ. Lower values create a sharper distribution.

    Returns:
        Scalar loss value.
    """
    # Ensure unit vectors
    z_i = F.normalize(z_i, p=2, dim=1)
    z_j = F.normalize(z_j, p=2, dim=1)

    N = z_i.size(0)

    # All-pairs cosine similarity matrix  (N × N)
    sim = torch.matmul(z_i, z_j.T) / temperature   # [N, N]

    # Positive pairs are the diagonal; labels index into rows of sim
    labels = torch.arange(N, device=z_i.device)

    # Numerically stable cross-entropy (subtracting row max)
    logits = sim - sim.max(dim=1, keepdim=True).values.detach()
    loss = F.cross_entropy(logits, labels)

    return loss


# Alias for backwards compatibility with existing scripts
subject_contrastive_loss = nt_xent_loss
