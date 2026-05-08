# Processed Splits

This folder is the target location for the generated MB1 and MB2 window splits.

The full processed splits are **not** bundled in the repository because the
generated `train.npz`, `val.npz`, and `test.npz` files can be large and may
exceed practical GitHub limits. Instead, the repository includes:

- the preprocessing and split-generation code,
- small sample `.npz` files for quick checks,
- the fixed sampled-test CSVs used for status evaluation.

## How To Generate Full Splits

From the repository root:

```bash
python scripts/01_data_preparation/make_temporal_splits.py --config configs/default_mb1.yaml
python scripts/01_data_preparation/make_temporal_splits.py --config configs/default_mb2.yaml
```

If you start from the original raw source logs instead of already filtered
per-model CSVs, run this first:

```bash
python scripts/01_data_preparation/preprocess_raw_logs.py --config configs/default_mb2.yaml
```

## Expected Output Layout

```text
data/splits/MB1_round1/train.npz
data/splits/MB1_round1/val.npz
data/splits/MB1_round1/test.npz
data/splits/MB1_round2/train.npz
...
data/splits/MB2_round3/test.npz
```

Each split file is an `.npz` bundle containing:

- `X`: window tensor shaped `[num_windows, window_days, num_attributes]`
- `y`: binary label (`0=healthy`, `1=failed/at risk`)
- `ttf`: time-to-failure in days
- `features`: SMART attribute names

## Fixed Sampled Test Set

After the full test splits are created, build the fixed imbalanced evaluation
subset with:

```bash
python scripts/01_data_preparation/make_imbalanced_test_set.py --config configs/default_mb2.yaml
```

This writes:

- `data/splits/sampled_test_1to23/sampled_test_indices.csv`
- `data/splits/sampled_test_1to23/sampling_summary.csv`
