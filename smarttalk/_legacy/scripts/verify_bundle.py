#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

REQUIRED_PATHS = [
    ROOT / "README.md",
    ROOT / "MISSING.md",
    ROOT / "requirements.txt",
    ROOT / "papers" / "SMARTTalk_OSDI_26.pdf",
    ROOT / "papers" / "osdi26-paper2425-supplementary_material.pdf",
    ROOT / "configs" / "paper_tables" / "table5_status.csv",
    ROOT / "configs" / "paper_tables" / "table6_ttf.csv",
    ROOT / "configs" / "paper_tables" / "table7_explanations.csv",
    ROOT / "configs" / "llm_backbones.json",
    ROOT / "code" / "core" / "n_day_window.py",
    ROOT / "code" / "core" / "step1.py",
    ROOT / "code" / "core" / "learn_vocab_from_prototypes.py",
    ROOT / "code" / "core" / "raw_llm_eval.py",
    ROOT / "code" / "core" / "heuristic_llm_eval.py",
    ROOT / "code" / "core" / "llm_eval.py",
    ROOT / "code" / "core" / "mb2_dump_summaries.py",
    ROOT / "code" / "core" / "filter_dataset.py",
    ROOT / "code" / "nl_eval" / "exp_rec_generation.py",
    ROOT / "code" / "nl_eval" / "perturbation_eval.py",
    ROOT / "code" / "nl_eval" / "judge_explanations.py",
    ROOT / "scripts" / "build_processed_splits.py",
    ROOT / "scripts" / "build_offline_artifacts.py",
    ROOT / "scripts" / "generate_prototype_figures.py",
    ROOT / "scripts" / "run_table56_evals.py",
    ROOT / "scripts" / "run_single_model_inference.sh",
    ROOT / "scripts" / "serve_vllm.sh",
    ROOT / "scripts" / "aggregate_table56_metrics.py",
    ROOT / "scripts" / "run_table7_pipeline.py",
    ROOT / "scripts" / "aggregate_table7_metrics.py",
    ROOT / "prompts" / "raw_system_prompt.txt",
    ROOT / "prompts" / "heuristic_system_prompt.txt",
    ROOT / "prompts" / "smarttalk_system_prompt.txt",
    ROOT / "prompts" / "judge_prompt_template.txt",
    ROOT / "data" / "artifacts" / "MB1_round1" / "mb1_prototypes_with_phrases.npz",
    ROOT / "data" / "artifacts" / "MB2_round1" / "mb2_prototypes_with_phrases.npz",
    ROOT / "data" / "processed" / "MB1_round1" / "test.npz",
    ROOT / "data" / "processed" / "MB2_round1" / "test.npz",
    ROOT / "data" / "raw" / "dataset_by_model" / "MB1",
    ROOT / "data" / "raw" / "dataset_by_model" / "MB2",
]


def main() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED_PATHS if not path.exists()]
    payload = {
        "root": str(ROOT),
        "required_count": len(REQUIRED_PATHS),
        "missing": missing,
    }

    print(json.dumps(payload, indent=2))

    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
