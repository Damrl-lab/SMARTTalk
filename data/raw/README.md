# Raw Data Placement

This folder is intentionally mostly empty in the GitHub artifact.

The full Alibaba SMART dataset is public, but the raw files are too large to
bundle directly in this repository. Download the source data from:

- <https://tianchi.aliyun.com/dataset/95044>

Then place the raw inputs in one of these layouts:

## Option A: original source logs

```text
data/raw/source_logs/
```

Run:

```bash
python scripts/01_data_preparation/preprocess_raw_logs.py --config configs/default_mb2.yaml
```

This creates the filtered per-model daily SMART CSVs used by the next stage.

## Option B: already filtered per-model CSVs

```text
data/raw/dataset_by_model/MB1/
data/raw/dataset_by_model/MB2/
data/raw/ssd_failure_tag.csv
```

Run:

```bash
python scripts/01_data_preparation/make_temporal_splits.py --config configs/default_mb2.yaml
```

That command generates the processed train/val/test `.npz` splits under
`data/splits/`.
