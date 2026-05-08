#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 scripts/verify_bundle.py
python3 scripts/generate_paper_results.py
python3 scripts/export_phrase_dictionary_stats.py

echo
echo "Offline bundle outputs refreshed."
echo "For Figure 4 / Figure 5 regeneration, run:"
echo "  python3 scripts/generate_prototype_figures.py --dataset-name MB2 --round 1 --device cpu"
echo
echo "For live Table 5 / Table 6 LLM runs, use:"
echo "  python3 scripts/run_table56_evals.py ..."
echo
echo "For live Table 7 judge + perturbation runs, use:"
echo "  python3 scripts/run_table7_pipeline.py --judge-model-name <judge_model_id>"
