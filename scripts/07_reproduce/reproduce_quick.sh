#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python scripts/01_data_preparation/prepare_sample_data.py
python scripts/01_data_preparation/verify_data_schema.py --config configs/default_mb2.yaml
python scripts/05_evaluation/make_table5_status.py --config configs/default_mb2.yaml
python scripts/05_evaluation/make_table6_ttf.py --config configs/default_mb2.yaml
