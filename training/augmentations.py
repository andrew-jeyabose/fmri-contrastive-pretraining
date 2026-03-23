"""
training/augmentations.py
--------------------------
Domain-specific fMRI augmentations for contrastive pre-training.

Spatial augmentations act on 3D ICA maps (numpy arrays or tensors).
Temporal augmentations act on 1D ICA timeseries.

Design principle
~~~~~~~~~~~~~~~~
When augmenting a positive pair (pre, post) we apply the *same random
parameters* to both views so that the model must distinguish genuine
pre-vs-post differences rather than augmentation artefacts.

Usage
~~~~~
    aug_pre, aug_post = augment_spatial_pair([pre_vol, post_vol])
    aug_ts_pre  = augment_temporal(pre_ts)
    aug_ts_post = augment_temporal(post_ts)
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import rotate


# ---------------------------------------------------------------------------
# Spatial augmentations (3D volumes)
# ---------------------------------------------------------------------------

def augment_spatial_pair(
    volumes: list[torch.Tensor],
    angle_range: float = 15.0,
    noise_std:   float = 0.01,
    flip_prob:   float = 0.5,
    device: torch.device | None = None,
) -> list[torch.Tensor]:
    """Apply consistent random augmentations to a list of 3D volumes.

    The same rotation angle, flip pattern and noise seed are applied to every
    volume in the list, preserving the correspondence between pre/post pairs.

    Args:
        volumes:     List of tensors, each shape (1, D, H, W) or (B, 1, D, H, W).
                     Batched inputs are processed sample-by-sample.
        angle_range: Maximum rotation angle in degrees (uniform ±angle_range).
        noise_std:   Standard deviation of additive Gaussian noise.
        flip_prob:   Independent probability of flipping each spatial axis.
        device:      Target device for output tensors (inferred from input if None).

    Returns:
        List of augmented tensors, same shapes as input.
    """
    # Sample shared augmentation parameters
    angle    = np.random.uniform(-angle_range, angle_range)
    flip_axes = [ax for ax in [1, 2] if np.random.rand() < flip_prob]

    if device is None:
        device = volumes[0].device

    augmented = []
    for vol in volumes:
        arr = vol.cpu().numpy()

        # Rotation in the axial (H, W) plane
        arr = rotate(arr, angle, axes=(2, 3) if arr.ndim == 5 else (1, 2), reshape=False)

        # Axis flips
        for ax in flip_axes:
            real_ax = ax + 1 if arr.ndim == 5 else ax   # account for batch dim
            arr = np.flip(arr, axis=real_ax)

        # Gaussian noise
        arr = arr + np.random.normal(0, noise_std, arr.shape)

        augmented.append(torch.tensor(arr.copy(), dtype=torch.float32).to(device))

    return augmented


# ---------------------------------------------------------------------------
# Temporal augmentations (1D timeseries)
# ---------------------------------------------------------------------------

def augment_temporal(
    ts: torch.Tensor,
    mask_ratio:    float = 0.15,
    amplitude_range: tuple[float, float] = (0.85, 1.15),
    phase_shift_max: int = 5,
) -> torch.Tensor:
    """Apply random augmentations to a 1D ICA timeseries.

    Three independent augmentations are applied:

    1. **Temporal masking** — a random contiguous segment (``mask_ratio`` of
       total length) is zeroed out, simulating missing acquisitions or motion
       artefacts.

    2. **Amplitude jitter** — the entire signal is scaled by a random factor
       drawn uniformly from ``amplitude_range``, reflecting global signal
       variability across sessions.

    3. **Regional phase shift** — a random sub-segment is cyclically shifted
       by ±``phase_shift_max`` timepoints, mimicking slight timing offsets
       between haemodynamic responses.

    Args:
        ts:              Tensor of shape (1, T) or (B, 1, T).
        mask_ratio:      Fraction of timepoints to mask (0 → no masking).
        amplitude_range: (min, max) multiplier for amplitude jitter.
        phase_shift_max: Maximum phase shift in timepoints.

    Returns:
        Augmented tensor, same shape as input.
    """
    arr = ts.cpu().numpy().copy()
    T   = arr.shape[-1]

    # 1. Temporal masking
    mask_len   = max(1, int(T * mask_ratio))
    mask_start = np.random.randint(0, T - mask_len)
    arr[..., mask_start : mask_start + mask_len] = 0.0

    # 2. Amplitude jitter
    scale = np.random.uniform(*amplitude_range)
    arr   = arr * scale

    # 3. Regional phase shift
    if phase_shift_max > 0:
        seg_len   = T // 4
        seg_start = np.random.randint(0, T - seg_len)
        shift     = np.random.randint(-phase_shift_max, phase_shift_max + 1)
        arr[..., seg_start : seg_start + seg_len] = np.roll(
            arr[..., seg_start : seg_start + seg_len], shift, axis=-1
        )

    return torch.tensor(arr, dtype=torch.float32).to(ts.device)
