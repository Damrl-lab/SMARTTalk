# DATA_ACCESS

## Public Dataset Source

The complete Alibaba SSD SMART dataset used by SMARTTalk can be downloaded from:

- <https://tianchi.aliyun.com/dataset/95044>

## What This Repository Includes

This repository includes:

- small sample `.npz` windows in `data/sample_data/`,
- cached tables and figures,
- learned checkpoints and phrase-dictionary artifacts,
- scripts for reconstructing processed windows, sampled test sets, and ablations.

The full raw public dataset is not included here because the source files are
large.

## Expected Raw Data Placement

After download, place the raw inputs in one of the following layouts:

### Option A: original yearly raw logs

Place the extracted source logs under:

```text
data/raw/source_logs/
```

Then run:

```bash
python scripts/01_data_preparation/preprocess_raw_logs.py --config configs/default_mb1.yaml
```

This filtering step is shared by MB1 and MB2, so either default config works.

### Option B: already filtered per-model daily CSVs

Place per-model daily SMART CSVs under:

```text
data/raw/dataset_by_model/MB1/
data/raw/dataset_by_model/MB2/
```

and the failure metadata file at:

```text
data/raw/ssd_failure_tag.csv
```

The `ssd_failure_tag.csv` file comes from the Alibaba Tianchi dataset package
and should be copied into this location after download.

Then run:

```bash
python scripts/01_data_preparation/make_temporal_splits.py --config configs/default_mb1.yaml
python scripts/01_data_preparation/make_temporal_splits.py --config configs/default_mb2.yaml
```

## Processed Split Layout

The main training / inference code expects:

```text
data/splits/MB1_round1/train.npz
data/splits/MB1_round1/val.npz
data/splits/MB1_round1/test.npz
...
data/splits/MB2_round3/test.npz
```

Each `.npz` contains:

- `X`: `[N, window_days, num_attributes]`
- `y`: binary health label (`0=healthy`, `1=failed/at risk`)
- `ttf`: time-to-failure in days
- `features`: ordered SMART attribute names

## Fixed Imbalanced Test Set

For status evaluation, SMARTTalk uses all failed windows plus sampled healthy
windows at a fixed `1:23` ratio. Build it with:

```bash
python scripts/01_data_preparation/make_imbalanced_test_set.py --config configs/default_mb1.yaml
```

This command combines MB1 and MB2, so either default config works.

This writes:

- `data/splits/sampled_test_1to23/sampled_test_indices.csv`
- `data/splits/sampled_test_1to23/sampling_summary.csv`

## Sample Data

Bundled quick-run samples are provided in:

- `data/sample_data/MB1_round1_test_sample.npz`
- `data/sample_data/MB2_round1_test_sample.npz`

These are intentionally small and only meant for smoke tests, schema checks, and
package verification.
