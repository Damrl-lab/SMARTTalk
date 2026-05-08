#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python scripts/05_evaluation/make_table5_status.py --config configs/default_mb2.yaml
python scripts/05_evaluation/make_table6_ttf.py --config configs/default_mb2.yaml
python scripts/05_evaluation/make_table7_explanations.py --config configs/default_mb2.yaml
python scripts/06_ablation/run_ablation_from_cache.py
