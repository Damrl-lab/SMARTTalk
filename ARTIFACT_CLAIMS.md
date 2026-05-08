# ARTIFACT_CLAIMS

This file maps major paper claims to artifact modules, runnable commands, and
expected outputs.

## Claim 1: SMARTTalk improves status prediction over Raw-LLM and Heuristic-LLM

- Paper evidence: Table 5 in `paper/SMARTTalk_OSDI_Final.pdf`
- Commands:
  - `python scripts/05_evaluation/make_table5_status.py --config configs/default_mb2.yaml`
  - `python scripts/03_baselines/run_raw_llm.py --config configs/default_mb2.yaml`
  - `python scripts/03_baselines/run_heuristic_llm.py --config configs/default_mb2.yaml`
  - `python scripts/04_inference/run_smarttalk_inference.py --config configs/default_mb2.yaml`
- Main outputs:
  - `results/tables/table5_status_with_fpr_fnr.csv`
  - `results/tables/status_sampled_1to23/status_metrics_with_fpr_fnr.csv`

## Claim 2: SMARTTalk yields useful TTF bucket predictions

- Paper evidence: Table 6
- Commands:
  - `python scripts/05_evaluation/make_table6_ttf.py --config configs/default_mb2.yaml`
- Outputs:
  - `results/tables/table6_ttf.csv`

## Claim 3: SMARTTalk produces strong explanations and recommendations

- Paper evidence: Table 7
- Commands:
  - `python scripts/05_evaluation/make_table7_explanations.py --config configs/default_mb2.yaml`
- Outputs:
  - `results/tables/table7_explanations.csv`

## Claim 4: Figure 3 offline pattern learning is reproducible

- Artifact modules:
  - `smarttalk/patterns/`
  - `scripts/02_offline_pattern_learning/run_offline_pipeline.py`
- Command:
  - `python scripts/02_offline_pattern_learning/run_offline_pipeline.py --config configs/default_mb2.yaml`
- Outputs:
  - `artifacts/checkpoints/`
  - `artifacts/pattern_memory/`
  - `artifacts/phrase_dictionaries/`

## Claim 5: Figure 4 / Figure 5 style prototype visualizations can be regenerated

- Commands:
  - `python smarttalk/_legacy/scripts/generate_prototype_figures.py --dataset-name MB2 --round 1 --device cpu --output-root results/figures/paper_figures`
- Outputs:
  - `results/figures/paper_figures/fig_attr_prototypes.png`
  - `results/figures/paper_figures/fig_cross_prototypes.png`

## Claim 6: Rebuttal ablations support N/L sensitivity analysis

- Commands:
  - `python scripts/06_ablation/run_ablation_N.py --config configs/ablation_N.yaml`
  - `python scripts/06_ablation/run_ablation_L.py --config configs/ablation_L.yaml`
  - `python scripts/06_ablation/run_ablation_from_cache.py --config configs/ablation_N.yaml`
- Outputs:
  - `results/figures/ablation/main.png`
  - `results/figures/ablation/metrics_mb1.png`
  - `results/figures/ablation/metrics_mb2.png`

## Claim 7: Cached reproduction works without live LLM access

- Command:
  - `bash scripts/07_reproduce/reproduce_from_cache.sh`
- Outputs:
  - `results/tables/`
  - `results/figures/`

## Claim 8: Full raw-data reproduction is available when compute is provided

- Commands:
  - `bash scripts/07_reproduce/reproduce_full.sh`
- Requirements:
  - raw dataset from Alibaba Tianchi
  - GPU resources for CNN training
  - optional local vLLM or API credentials for live LLM inference
