#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python scripts/01_data_preparation/make_temporal_splits.py --config configs/default_mb2.yaml
python scripts/01_data_preparation/make_imbalanced_test_set.py --config configs/default_mb2.yaml
python scripts/02_offline_pattern_learning/run_offline_pipeline.py --config configs/default_mb2.yaml
python scripts/03_baselines/train_baseline.py --config configs/default_mb2.yaml --model rf
python scripts/04_inference/run_smarttalk_inference.py --config configs/default_mb2.yaml
python scripts/05_evaluation/make_table5_status.py --config configs/default_mb2.yaml
python scripts/05_evaluation/make_table6_ttf.py --config configs/default_mb2.yaml
python scripts/05_evaluation/make_table7_explanations.py --config configs/default_mb2.yaml
