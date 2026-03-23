# Data Format

The dataset used in this project is private clinical fMRI data and cannot be distributed.
This file documents the required format so you can adapt the pipeline to your own data.

## Expected CSV schema

The main CSV file (set via `configs/default.yaml → data.csv_path`) must contain the
following columns:

| Column                     | Type   | Description |
|----------------------------|--------|-------------|
| `subject_id`               | string | Participant identifier, e.g. `sub-01` |
| `network`                  | string | Functional network label: `Motor`, `Vision`, `Language`, `Frontal`, `Temporal` |
| `condition`                | string | `pre` or `post` surgery |
| `aggregated_spatial_path`  | string | Absolute path to NIfTI ICA spatial map (`.nii` or `.nii.gz`) |
| `aggregated_temporal_path` | string | Absolute path to text file containing ICA timeseries (one value per row) |

For the downstream classifier, an additional column is required:

| Column   | Type | Description                   |
|----------|------|-------------------------------|
| `label`  | int  | 0 = pre-surgery, 1 = post-surgery |

## Spatial maps

- Format: NIfTI (`.nii` or `.nii.gz`)
- Expected shape after loading: `(H, W, D)` — the dataset class transposes to `(D, H, W)`
- The encoder assumes spatial dimensions that reduce to `(6, 6, 5)` after 3× MaxPool3d(2).
  If your volumes differ, update `SpatialBranch.flat_dim` in `model/encoder.py`.

## Temporal timeseries

- Format: plain text, one floating-point value per line
- The dataset class resamples all timeseries to `temporal_input_dim` (default 590)
  using `scipy.signal.resample` if the native length differs.

## Generating ICA components

The spatial maps and timeseries should be ICA component outputs from a tool such as
**FSL MELODIC** or **ICA-AROMA**.  Each row in the CSV corresponds to one ICA component
assigned to a functional network via dual-regression or manual labelling.

## Directory structure suggestion

```
data/
├── sub-01/
│   ├── pre/
│   │   ├── Motor_spatial.nii.gz
│   │   └── Motor_temporal.txt
│   └── post/
│       ├── Motor_spatial.nii.gz
│       └── Motor_temporal.txt
├── sub-02/
│   └── ...
└── metadata.csv      ← the CSV file referenced above
```
