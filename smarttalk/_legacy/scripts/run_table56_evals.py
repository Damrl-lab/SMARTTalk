#!/usr/bin/env python3
"""
Run the Raw-LLM, Heuristic-LLM, and SMARTTalk evaluators for Table 5 and Table 6.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_SAMPLED_INDICES = "data/processed/sampled_test_1to23/sampled_test_indices.csv"


SCRIPT_MAP = {
    "raw": "code/core/raw_llm_eval.py",
    "heuristic": "code/core/heuristic_llm_eval.py",
    "smarttalk": "code/core/llm_eval.py",
}


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute the Table 5 / Table 6 LLM evaluations across backbones and rounds.",
    )
    parser.add_argument("--methods", nargs="+", default=["raw", "heuristic", "smarttalk"],
                        choices=["raw", "heuristic", "smarttalk"],
                        help="Which evaluation methods to run.")
    parser.add_argument("--backbones", nargs="+", default=["OS1", "OS2", "OS3", "OS4", "PROP"],
                        help="Backbone codes from configs/llm_backbones.json.")
    parser.add_argument("--datasets", nargs="+", default=["MB1", "MB2"],
                        choices=["MB1", "MB2"],
                        help="Datasets to evaluate.")
    parser.add_argument("--rounds", nargs="+", type=int, default=[1, 2, 3],
                        help="Temporal rounds to evaluate.")
    parser.add_argument("--config-path", type=str, default="configs/llm_backbones.json",
                        help="Backbone config JSON.")
    parser.add_argument("--run-root", type=str, default="results/llm_runs/table56",
                        help="Output root for per-run artifacts.")
    parser.add_argument("--processed-root", type=str, default="data/processed",
                        help="Root containing dataset round processed folders.")
    parser.add_argument("--artifacts-root", type=str, default="data/artifacts",
                        help="Root containing dataset round SMARTTalk artifact folders.")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Base URL for open-source backbones. Defaults to config file.")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for open-source backbones. Defaults to OPENAI_API_KEY or EMPTY.")
    parser.add_argument("--prop-base-url", type=str, default=None,
                        help="Optional override for the proprietary backbone endpoint.")
    parser.add_argument("--prop-api-key", type=str, default=None,
                        help="Optional override for the proprietary backbone API key.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Optional cap on the number of failed windows per dataset/round when --evaluate-all is not set.")
    parser.add_argument("--sample-seed", type=int, default=2026,
                        help="Sampling seed for random healthy-window selection.")
    parser.add_argument("--healthy-per-fail", type=float, default=23.0,
                        help="Number of healthy windows to sample per failed window.")
    parser.add_argument("--sampled-indices-csv", type=str, default=DEFAULT_SAMPLED_INDICES,
                        help="Optional fixed sampled-test CSV shared by all methods.")
    parser.add_argument("--evaluate-all", action="store_true",
                        help="Evaluate the full test split instead of the sampled subset.")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip model calls and only aggregate existing run outputs.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    configs = json.loads((root / args.config_path).read_text(encoding="utf-8"))
    run_root = root / args.run_root
    run_root.mkdir(parents=True, exist_ok=True)

    if not args.aggregate_only:
        for method in args.methods:
            script_path = SCRIPT_MAP[method]
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
                        out_dir = run_root / method / backbone
                        out_dir.mkdir(parents=True, exist_ok=True)
                        metrics_json = out_dir / f"{dataset}_round{round_id}_metrics.json"
                        predictions_jsonl = out_dir / f"{dataset}_round{round_id}_predictions.jsonl"
                        cmd = [
                            sys.executable,
                            script_path,
                            "--dataset-name", dataset,
                            "--round", str(round_id),
                            "--processed-root", args.processed_root,
                            "--model-name", model_name,
                            "--base-url", base_url,
                            "--api-key", api_key,
                            "--temperature", str(args.temperature),
                            "--max-tokens", str(args.max_tokens),
                            "--sample-seed", str(args.sample_seed),
                            "--healthy-per-fail", str(args.healthy_per_fail),
                            "--output-jsonl", str(predictions_jsonl),
                            "--output-metrics-json", str(metrics_json),
                        ]
                        if args.sampled_indices_csv:
                            cmd.extend(["--sampled-indices-csv", args.sampled_indices_csv])
                        if args.num_samples is not None:
                            cmd.extend(["--num-samples", str(args.num_samples)])
                        if args.evaluate_all:
                            cmd.append("--evaluate-all")
                        if method == "smarttalk":
                            cmd.extend(["--artifact-root", args.artifacts_root])
                            tp_csv = out_dir / f"{dataset}_round{round_id}_tp.csv"
                            cmd.extend(["--output-tp-csv", str(tp_csv)])
                        run(cmd, cwd=root)

    run(
        [
            sys.executable,
            "scripts/aggregate_table56_metrics.py",
            "--run-root", str(run_root),
            "--output-dir", str(run_root / "aggregated"),
        ],
        cwd=root,
    )


if __name__ == "__main__":
    main()
