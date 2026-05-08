# SMARTTalk: Teaching SMART Logs to Talk to LLM

This repository is the artifact bundle for the paper *SMARTTalk: Teaching SMART
Logs to Talk to LLM*.

It contains the code, configs, sample data, cached outputs, and documentation
needed to reproduce the main experiments, rebuild the processed SMART windows
from the public Alibaba dataset, and inspect the learned pattern-memory and
phrase-dictionary artifacts.

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

This checks the package structure, validates the sample `.npz` files, and
regenerates the paper-facing tables from cached outputs.

### 2. Cached Reproduction

Regenerate the main paper tables and figures without API keys:

```bash
cd <repo-root>
bash scripts/07_reproduce/reproduce_from_cache.sh
```

This uses the bundled table snapshots, cached phrase-dictionary outputs,
sampled-test results, and cached ablation figures.

### 3. Full Reproduction

Run preprocessing, offline pattern learning, inference, evaluation, and
ablation using the full dataset and available compute:

```bash
cd <repo-root>
bash scripts/07_reproduce/reproduce_full.sh
```

Full reproduction assumes:

- the Alibaba SSD SMART dataset has been downloaded,
- raw data is placed as described in `DATA_ACCESS.md`,
- GPU resources are available for CNN training and optional local vLLM serving,
- API keys or local model endpoints are configured if live LLM inference is used.

## Common Commands

### Prepare full MB1 / MB2 temporal splits

```bash
python scripts/01_data_preparation/make_temporal_splits.py --config configs/default_mb2.yaml
python scripts/01_data_preparation/make_imbalanced_test_set.py --config configs/default_mb2.yaml
```

### Rebuild offline SMARTTalk artifacts

```bash
python scripts/02_offline_pattern_learning/run_offline_pipeline.py --config configs/default_mb2.yaml
```

### Run one baseline or LLM method

```bash
python scripts/03_baselines/train_baseline.py --config configs/default_mb2.yaml --model rf
python scripts/03_baselines/run_raw_llm.py --config configs/default_mb2.yaml
python scripts/03_baselines/run_heuristic_llm.py --config configs/default_mb2.yaml
python scripts/04_inference/run_smarttalk_inference.py --config configs/default_mb2.yaml
```

### Regenerate paper tables

```bash
python scripts/05_evaluation/make_table5_status.py --config configs/default_mb2.yaml
python scripts/05_evaluation/make_table6_ttf.py --config configs/default_mb2.yaml
python scripts/05_evaluation/make_table7_explanations.py --config configs/default_mb2.yaml
```

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
does **not** bundle the full raw dataset. See `DATA_ACCESS.md` and `data/README.md`
for placement and preprocessing details.

The full generated `train.npz`, `val.npz`, and `test.npz` split trees are also
left out of the repository by design, because those processed files can exceed
typical GitHub-friendly sizes. The preprocessing and split-generation code is
included so evaluators can rebuild them locally from the public raw dataset.

## What Is Bundled

- paper tables and figures,
- checkpoints and phrase-dictionary artifacts needed for inspection,
- cached Table 5 sampled-set outputs with FPR/FNR,
- cached ablation figures and supporting CSV summaries,
- small sample `.npz` splits for smoke tests.

## What Must Be Supplied Externally

- the full Alibaba raw SMART dataset,
- any project-approved public release location for very large processed splits if
  you choose not to regenerate them locally,
- live LLM endpoints or API keys for full online evaluation,
- a project-approved open-source license text for public release.

## Notes

- Positive class for status prediction is `RISK / failed`.
- The paper’s fixed imbalanced sampled test set uses `1 failed : 23 healthy`.
- The main paper setting is `N = 30` days and `L = 5` days.
- The sensitivity studies vary `N in {10,20,30,40,50}` and `L in {2,4,5,10,15}`.

See:

- `ARTIFACT_CLAIMS.md`
- `REPRODUCIBILITY.md`
- `ORGANIZATION_REPORT.md`
- `artifacts/MANIFEST.md`
