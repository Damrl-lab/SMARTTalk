# Processed Split Layout

The full processed window splits are **not** bundled in the GitHub artifact.

Reason:

- the generated `train.npz`, `val.npz`, and `test.npz` files can be large,
- GitHub has a hard per-file size limit,
- this repository therefore keeps only small sample `.npz` files and the fixed
  sampled-test CSVs needed for quick verification.

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
