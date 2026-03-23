# fMRI Contrastive Pre-training for Surgical Outcome Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A self-supervised contrastive learning framework for resting-state fMRI that learns
subject-invariant network representations, enabling few-shot transfer to pre/post surgical
outcome classification with limited labelled data.

---

## Motivation

Resting-state fMRI (rs-fMRI) ICA decompositions capture both a **spatial map** (where a
network lives in the brain) and a **temporal signal** (how it fluctuates over time).
Supervised training on these is bottlenecked by small clinical cohorts.

This project asks: *can we learn rich, transferable embeddings from paired pre/post-surgery
scans without labels, then fine-tune a lightweight classifier on top?*

We treat each subject's **pre-surgery** and **post-surgery** scan of the same functional
network as a **natural positive pair** — they share subject identity and network identity
but differ in post-operative brain state. The model must learn to pull these pairs together
in embedding space while pushing apart different subjects, capturing what changes (surgical
effect) and what is stable (network identity).

---

## Architecture

```
           Spatial Branch                Temporal Branch
      ┌──────────────────────┐      ┌──────────────────────┐
      │  3D ICA map (NIfTI)  │      │  1D timeseries (TXT) │
      │  (1, D, H, W)        │      │  (1, T)              │
      └──────────┬───────────┘      └──────────┬───────────┘
                 │                              │
         Conv3D × 6                      Conv1D × 2
         MaxPool3D × 3                   MaxPool1D × 2
                 │                              │
            FC → 256                       FC → 128
                 └──────────────┬─────────────┘
                                │  cat([256, 128])
                           FC → 128  (embedding)
                                │
                     ┌──────────┴──────────┐
                     │   Projection Head    │
                     │  128 → 64 → 32 (L2) │
                     └─────────────────────┘
                                │
                     NT-Xent (SimCLR) Loss
                     on pre/post pairs
```

### Key design choices

| Component | Detail |
|---|---|
| Spatial backbone | 3D CNN (6 conv layers, 3 maxpool) → 256-d |
| Temporal backbone | 1D CNN (2 conv layers, 2 maxpool) → 128-d |
| Joint embedding | 384-d → 128-d via FC |
| Projection head | MLP 128 → 64 → 32, L2-normalised |
| Loss | NT-Xent (temperature τ = 0.05) |
| Positive pair | Same subject, same network, pre vs post surgery |
| Augmentations | Random 3D rotation, axis flips, Gaussian noise (spatial); temporal masking, amplitude jitter (temporal) |

---

## Augmentation Strategy

Domain-specific augmentations are critical for fMRI. We apply:

**Spatial (3D ICA maps)**
- Random rotation ±15° in the axial plane
- Random axis flips (left-right, anterior-posterior)
- Gaussian noise σ = 0.01

**Temporal (ICA timeseries)**
- Random temporal masking (10–20% of timepoints zeroed)
- Amplitude jitter (scale ∈ [0.8, 1.2])
- Regional perturbation: slight phase shift of segments

These are applied identically to pre and post spatial maps within a pair so the model
learns differences driven by surgical effect rather than augmentation artefacts.

---

## Repository Structure

```
fmri-contrastive-pretraining/
├── model/
│   ├── encoder.py            # Dual-branch spatiotemporal CNN encoder
│   └── projection_head.py    # MLP projection head
├── training/
│   ├── contrastive_loss.py   # NT-Xent / SimCLR loss
│   ├── augmentations.py      # fMRI-specific augmentations
│   ├── dataset.py            # Dataset & DataLoader
│   └── train.py              # Pre-training loop
├── downstream/
│   └── classifier.py         # Fine-tuning classifier (frozen encoder)
├── configs/
│   └── default.yaml          # All hyperparameters
├── scripts/
│   └── run_pretraining.sh    # SLURM / local launch script
├── notebooks/
│   └── visualise_embeddings.ipynb  # t-SNE / UMAP of learned embeddings
├── data/
│   └── README.md             # Dataset format specification
├── requirements.txt
└── README.md
```

---

## Data Format

The model expects a CSV with columns:

| Column | Description |
|---|---|
| `subject_id` | Participant identifier (e.g. `sub-01`) |
| `network` | Functional network label (`Motor`, `Vision`, `Language`, `Frontal`, `Temporal`) |
| `condition` | `pre` or `post` surgery |
| `aggregated_spatial_path` | Path to NIfTI ICA spatial map (`.nii.gz`) |
| `aggregated_temporal_path` | Path to temporal ICA timeseries (`.txt`) |
| `label` | 0 = pre, 1 = post (used in fine-tuning only) |

The clinical dataset used in this work is private and cannot be shared. See
`data/README.md` for full format specification and instructions for adapting the
pipeline to your own rs-fMRI ICA outputs (e.g. from FSL MELODIC).

---

## Installation

```bash
git clone https://github.com/<your-username>/fmri-contrastive-pretraining.git
cd fmri-contrastive-pretraining
pip install -r requirements.txt
```

Requires Python ≥ 3.9, PyTorch ≥ 2.0, CUDA recommended.

---

## Running Pre-training

```bash
# Edit configs/default.yaml to set your CSV path and output directory
python training/train.py --config configs/default.yaml
```

Or with the SLURM script:

```bash
sbatch scripts/run_pretraining.sh
```

### Key config options

```yaml
data:
  csv_path: "path/to/your_data.csv"
  temporal_input_dim: 590        # number of fMRI timepoints

model:
  embedding_dim: 128
  projection_dim: 32

training:
  batch_size: 16
  epochs: 150
  lr: 1.0e-4
  temperature: 0.05
  early_stopping_patience: 20
```

---

## Fine-tuning for Surgical Outcome Classification

After pre-training, freeze the encoder and train a lightweight classifier:

```bash
python downstream/classifier.py \
  --encoder_weights outputs/best_encoder.pth \
  --csv path/to/labelled_data.csv \
  --network Motor \
  --n_folds 5
```

The downstream classifier uses 5-fold subject-stratified cross-validation and reports
accuracy, sensitivity, specificity, F1, and AUC per functional network.

---

## Results Summary

Pre-training on paired pre/post rs-fMRI ICA components (n ≈ XX subjects, XX networks)
and fine-tuning with 5-fold cross-validation:

| Network  | AUC  | F1   | Sensitivity | Specificity |
|----------|------|------|-------------|-------------|
| Motor    | —    | —    | —           | —           |
| Vision   | —    | —    | —           | —           |
| Language | —    | —    | —           | —           |
| Frontal  | —    | —    | —           | —           |
| Temporal | —    | —    | —           | —           |

> ⚠️ Results table will be updated upon paper acceptance. Placeholder rows are intentional —
> populate with your numbers before making the repo public.

---

## Citation

If you use this code, please cite:

```bibtex
@article{yourname2025fmricontrastive,
  title   = {Self-Supervised Contrastive Learning of Functional Brain Network Representations for Surgical Outcome Prediction},
  author  = {Your Name and Co-authors},
  journal = {Under Review},
  year    = {2025}
}
```

---

## License

MIT License. See `LICENSE` for details.

The dataset used for training is private/clinical and is not distributed with this repository.
Model weights trained on the clinical data are also not publicly released.
