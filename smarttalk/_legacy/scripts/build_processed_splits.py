#!/usr/bin/env python3
"""
Build train/val/test SMART windows from per-model daily SMART CSVs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MODEL_CODE = {"MB1": "B1", "MB2": "B2"}


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run code/core/n_day_window.py for MB1/MB2 temporal rounds.",
    )
    parser.add_argument("--dataset-name", type=str, default="MB2",
                        choices=["MB1", "MB2", "ALL"],
                        help="Which dataset to process.")
    parser.add_argument("--rounds", nargs="+", type=int, default=[1, 2, 3],
                        help="Temporal rounds to build.")
    parser.add_argument("--dataset-by-model-root", type=str, default="data/raw/dataset_by_model",
                        help="Folder containing per-model daily SMART CSVs.")
    parser.add_argument("--failure-tag-path", type=str, default="data/raw/ssd_failure_tag.csv",
                        help="Path to ssd_failure_tag.csv.")
    parser.add_argument("--processed-root", type=str, default="data/processed",
                        help="Output root for processed train/val/test splits.")
    parser.add_argument("--window-size", type=int, default=30,
                        help="Observation window size in days.")
    parser.add_argument("--fail-horizon-days", type=int, default=30,
                        help="Positive-label time-to-failure horizon in days.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    datasets = ["MB1", "MB2"] if args.dataset_name == "ALL" else [args.dataset_name]
    for dataset in datasets:
        for round_id in args.rounds:
            output_dir = Path(args.processed_root) / f"{dataset}_round{round_id}"
            cmd = [
                sys.executable,
                "code/core/n_day_window.py",
                "--model-folder-name", dataset,
                "--dataset-by-model-root", args.dataset_by_model_root,
                "--failure-tag-path", args.failure_tag_path,
                "--failure-model-value", MODEL_CODE[dataset],
                "--window-size", str(args.window_size),
                "--fail-horizon-days", str(args.fail_horizon_days),
                "--split-round", str(round_id),
                "--output-dir", str(output_dir),
            ]
            run(cmd, cwd=root)


if __name__ == "__main__":
    main()
