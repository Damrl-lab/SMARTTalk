# SMARTTalk: Teaching SMART Logs to Talk to LLM

This repository is the artifact bundle for the paper *SMARTTalk: Teaching SMART
Logs to Talk to LLM*.

It contains the code, configs, sample data, cached outputs, and documentation
needed to reproduce the main experiments, rebuild the processed SMART windows
from the public Alibaba dataset, and inspect the learned pattern-memory and
phrase-dictionary artifacts.

For evaluation with the bundled cached SMARTTalk artifacts, the repository also
includes the matching cached `test.npz` files under `data/splits/`. For a full
end-to-end rerun, the recommended workflow is to regenerate the processed
splits, rebuild the sampled test set, and rerun offline pattern learning before
launching live inference.

## Repository Layout

- `configs/`: default configs for MB1, MB2, LLM backbones, and ablations.
- `docker/`: Python requirements, Conda environment, and Dockerfile.
- `smarttalk/`: reusable package code and preserved low-level implementation files.
- `scripts/`: numbered CLI entry points matching the artifact workflow.
- `data/`: sample data, schema notes, and split-placement instructions.
- `artifacts/`: checkpoints, pattern-memory assets, phrase dictionaries, cached predictions, and cached ablation outputs.
- `results/`: paper tables, prototype figures, phrase-dictionary exports, and ablation figures.
- `tests/`: lightweight correctness and packaging tests.

## Reproduction Options

### 1. Quick Check

Use the bundled sample data and cached assets:

```bash
cd <repo-root>
bash scripts/07_reproduce/reproduce_quick.sh
```

This path uses only bundled sample data and cached outputs. It checks the
package structure, validates the sample `.npz` files, and regenerates the
paper-facing tables from cached state.

### 2. Cached Reproduction

Regenerate the main paper tables and figures without API keys:

```bash
cd <repo-root>
bash scripts/07_reproduce/reproduce_from_cache.sh
```

This path uses cached outputs only. It relies on the bundled table snapshots,
cached phrase-dictionary outputs, sampled-test results, cached ablation
figures, and the matching bundled `data/splits/*/test.npz` files used by the
cached evaluation path.

### 3. Full Reproduction

Run preprocessing, offline pattern learning, inference, evaluation, and
ablation using the full dataset and available compute:

```bash
cd <repo-root>
bash scripts/07_reproduce/reproduce_full.sh
```

Full reproduction assumes:

- the Alibaba SSD SMART dataset has been downloaded,
- raw data is placed under `data/raw/source_logs/` or `data/raw/dataset_by_model/`
  as described in `DATA_ACCESS.md`,
- GPU resources are available for CNN training and optional local vLLM serving,
- API keys or local model endpoints are configured if live LLM inference is used.

## Common Commands

### Prepare full MB1 / MB2 temporal splits

```bash
python scripts/01_data_preparation/preprocess_raw_logs.py --config configs/default_mb1.yaml
python scripts/01_data_preparation/make_temporal_splits.py --config configs/default_mb1.yaml
python scripts/01_data_preparation/make_temporal_splits.py --config configs/default_mb2.yaml
python scripts/01_data_preparation/make_imbalanced_test_set.py --config configs/default_mb1.yaml
```

The raw-log filtering step is shared by both datasets, so either default config
works there. The sampled-test builder scans the processed splits currently
present under `data/splits/`, so it can also be launched with either default
config.

The default raw-log location is `data/raw/source_logs/`, for example:

```text
data/raw/source_logs/
├── smartlog2018ssd/
└── smartlog2019ssd/
```

### Rebuild offline SMARTTalk artifacts

```bash
python scripts/02_offline_pattern_learning/run_offline_pipeline.py --config configs/default_mb1.yaml
python scripts/02_offline_pattern_learning/run_offline_pipeline.py --config configs/default_mb2.yaml
```

The default configs target round 1. To run another round, copy one of the
default configs and update its `round` field.

### Run one baseline or LLM method

```bash
python scripts/03_baselines/train_baseline.py --config configs/default_mb1.yaml --model rf
python scripts/03_baselines/train_baseline.py --config configs/default_mb2.yaml --model rf
python scripts/03_baselines/run_raw_llm.py --config configs/default_mb1.yaml
python scripts/03_baselines/run_heuristic_llm.py --config configs/default_mb1.yaml
python scripts/04_inference/run_smarttalk_inference.py --config configs/default_mb1.yaml
```

Use `configs/default_mb2.yaml` in the same commands when you want the MB2 run
instead of MB1.

### Regenerate paper tables

```bash
python scripts/05_evaluation/make_table5_status.py --config configs/default_mb1.yaml
python scripts/05_evaluation/make_table6_ttf.py --config configs/default_mb1.yaml
python scripts/05_evaluation/make_table7_explanations.py --config configs/default_mb1.yaml
```

These table-generation commands use shared cached outputs, so either default
config works here.

The paper-level Table 5 and Table 6 values are aggregated across rounds 1, 2,
and 3. The single-config convenience wrappers shown above are useful for local
one-dataset / one-round runs, while the paper-level reported values summarize
the aggregate over all three rounds.

### Run sensitivity studies

```bash
python scripts/06_ablation/run_ablation_N.py --config configs/ablation_N.yaml
python scripts/06_ablation/run_ablation_L.py --config configs/ablation_L.yaml
python scripts/06_ablation/run_ablation_from_cache.py --config configs/ablation_N.yaml
```

## Data Access

The full public dataset can be downloaded from Alibaba Tianchi:

- <https://tianchi.aliyun.com/dataset/95044>

This repository includes small sample `.npz` files for quick checks and the full
code path for rebuilding processed windows and the fixed imbalanced test set. It
does not bundle the full raw dataset. See `DATA_ACCESS.md` and `data/README.md`
for placement and preprocessing details.

The full generated `train.npz`, `val.npz`, and `test.npz` split trees are also
left out of the repository by design because those processed files are much
larger than the rest of the artifact. The repository may include matching
cached `test.npz` files for evaluation with the bundled artifacts, but the full
preprocessing and split-generation code is included so complete train/val/test
trees can be rebuilt locally from the public raw dataset.

## What Is Bundled

- paper tables and figures,
- checkpoints and phrase-dictionary artifacts needed for inspection,
- matching cached `test.npz` files for the bundled evaluation artifacts,
- cached Table 5 sampled-set outputs with FPR/FNR,
- cached ablation figures and supporting CSV summaries,
- small sample `.npz` splits for smoke tests.

## What Must Be Supplied Externally

- the full Alibaba raw SMART dataset,
- `data/raw/ssd_failure_tag.csv` from the Tianchi dataset package,
- the public Tianchi package may name this file `ssd_failure_label.csv`; the
  preprocessing scripts accept either filename,
- full `train.npz` and `val.npz` processed splits when running the complete
  pipeline without relying on the bundled cached evaluation files,
- live LLM endpoints or API keys for full online evaluation.

## Notes

- Positive class for status prediction is `RISK / failed`.
- The paper’s fixed imbalanced sampled test set uses `1 failed : 23 healthy`.
- The main paper setting is `N = 30` days and `L = 5` days.
- The sensitivity studies vary `N in {10,20,30,40,50}` and `L in {2,4,5,10,15}`.

See:

- `ARTIFACT_CLAIMS.md`
- `REPRODUCIBILITY.md`
- `artifacts/MANIFEST.md`

## Paper Reference

This artifact accompanies the following paper:

```bibtex
@inproceedings{AkewarEtAl_OSDI_2026,
  author    = {Akewar, Mayur and Luo, Dongsheng and Madireddy, Sandeep and Bhimani, Janki},
  title     = {SMARTTalk: Teaching SMART Logs to Talk to LLMs},
  booktitle = {20th USENIX Symposium on Operating Systems Design and Implementation (OSDI)},
  year      = {2026},
  note      = {To appear}
}
```
