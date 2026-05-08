#!/usr/bin/env python3
"""
Run the Table 7 judge and perturbation pipeline for SMARTTalk backbones.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute the Table 7 SMARTTalk judge and perturbation pipeline.",
    )
    parser.add_argument("--backbones", nargs="+", default=["OS1", "OS2", "OS3", "OS4", "PROP"])
    parser.add_argument("--datasets", nargs="+", default=["MB1", "MB2"], choices=["MB1", "MB2"])
    parser.add_argument("--rounds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--config-path", type=str, default="configs/llm_backbones.json")
    parser.add_argument("--table56-run-root", type=str, default="results/llm_runs/table56/smarttalk",
                        help="SMARTTalk prediction outputs from scripts/run_table56_evals.py.")
    parser.add_argument("--run-root", type=str, default="results/llm_runs/table7",
                        help="Output root for Table 7 run artifacts.")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Base URL for open-source SMARTTalk backbones.")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for open-source SMARTTalk backbones.")
    parser.add_argument("--prop-base-url", type=str, default=None,
                        help="Optional override for the proprietary backbone endpoint.")
    parser.add_argument("--prop-api-key", type=str, default=None,
                        help="Optional override for the proprietary backbone key.")
    parser.add_argument("--judge-model-name", type=str, required=True,
                        help="Judge model name. The paper used an external GPT-5.1 Thinking judge.")
    parser.add_argument("--judge-base-url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--judge-api-key", type=str, default=None)
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip new model calls and only aggregate existing run outputs.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    configs = json.loads((root / args.config_path).read_text(encoding="utf-8"))
    table56_root = root / args.table56_run_root
    run_root = root / args.run_root
    run_root.mkdir(parents=True, exist_ok=True)

    if not args.aggregate_only:
        judge_api_key = args.judge_api_key or os.environ.get("OPENAI_API_KEY")
        if not judge_api_key:
            raise RuntimeError("No judge API key provided. Pass --judge-api-key or set OPENAI_API_KEY.")

        for backbone in args.backbones:
            info = configs[backbone]
            model_name = info["model_name"]
            base_url = args.prop_base_url if backbone == "PROP" and args.prop_base_url else args.base_url or info["default_base_url"]
            api_key = (
                args.prop_api_key if backbone == "PROP" and args.prop_api_key is not None
                else args.api_key if args.api_key is not None
                else os.environ.get("OPENAI_API_KEY", "EMPTY")
            )
            for dataset in args.datasets:
                for round_id in args.rounds:
                    predictions = table56_root / backbone / f"{dataset}_round{round_id}_predictions.jsonl"
                    if predictions.exists():
                        judge_dir = run_root / "judge" / backbone
                        judge_dir.mkdir(parents=True, exist_ok=True)
                        run(
                            [
                                sys.executable,
                                "code/nl_eval/judge_explanations.py",
                                "--input-path", str(predictions),
                                "--output-csv", str(judge_dir / f"{dataset}_round{round_id}_judge.csv"),
                                "--output-metrics-json", str(judge_dir / f"{dataset}_round{round_id}_judge_metrics.json"),
                                "--filter-mode", "true_positive",
                                "--model-name", args.judge_model_name,
                                "--base-url", args.judge_base_url,
                                "--api-key", judge_api_key,
                            ],
                            cwd=root,
                        )

                    perturb_dir = run_root / "perturb" / backbone
                    perturb_dir.mkdir(parents=True, exist_ok=True)
                    run(
                        [
                            sys.executable,
                            "code/nl_eval/perturbation_eval.py",
                            "--dataset-name", dataset,
                            "--round", str(round_id),
                            "--model-name", model_name,
                            "--base-url", base_url,
                            "--api-key", api_key,
                            "--output-csv", str(perturb_dir / f"{dataset}_round{round_id}_perturb.csv"),
                            "--output-metrics-json", str(perturb_dir / f"{dataset}_round{round_id}_metrics.json"),
                        ],
                        cwd=root,
                    )

    run(
        [
            sys.executable,
            "scripts/aggregate_table7_metrics.py",
            "--run-root", str(run_root),
            "--output-csv", str(run_root / "aggregated" / "table7_from_runs.csv"),
        ],
        cwd=root,
    )


if __name__ == "__main__":
    main()
