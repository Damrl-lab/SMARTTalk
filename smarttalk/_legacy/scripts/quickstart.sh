#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 scripts/verify_bundle.py
python3 scripts/generate_paper_results.py
python3 scripts/export_phrase_dictionary_stats.py

echo "Outputs are available under:"
echo "  $ROOT/results/paper_tables"
echo "  $ROOT/results/paper_figures"
echo "  $ROOT/results/phrase_dictionary"
echo
echo "Additional entry points:"
echo "  $ROOT/scripts/build_processed_splits.py"
echo "  $ROOT/scripts/build_offline_artifacts.py"
echo "  $ROOT/scripts/run_table56_evals.py"
echo "  $ROOT/scripts/run_table7_pipeline.py"
