#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from sensitivity_common import (
    DEFAULT_BACKBONES,
    DEFAULT_DATASETS,
    DEFAULT_PATCH_METHODS,
    DEFAULT_PATCH_VALUES,
    DEFAULT_ROUNDS,
    DEFAULT_WINDOW_METHODS,
    DEFAULT_WINDOW_VALUES,
    FINALIZED_ROOT,
    ROOT,
    iter_settings,
)


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run N/L sensitivity Table 5 evaluations and generate aggregation plots.",
    )
    parser.add_argument("--study", choices=["window", "patch", "both"], default="both")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=["MB1", "MB2"])
    parser.add_argument("--rounds", nargs="+", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--window-values", nargs="+", type=int, default=DEFAULT_WINDOW_VALUES)
    parser.add_argument("--patch-values", nargs="+", type=int, default=DEFAULT_PATCH_VALUES)
    parser.add_argument("--backbones", nargs="+", default=DEFAULT_BACKBONES,
                        help="Representative backbones to evaluate. Defaults to OS3 + PROP.")
    parser.add_argument("--window-methods", nargs="+", default=DEFAULT_WINDOW_METHODS,
                        choices=["raw", "heuristic", "smarttalk"])
    parser.add_argument("--patch-methods", nargs="+", default=DEFAULT_PATCH_METHODS,
                        choices=["raw", "heuristic", "smarttalk"])
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--prop-base-url", type=str, default=None)
    parser.add_argument("--prop-api-key", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip model calls and only aggregate existing per-setting outputs.")
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()

    studies = ["window", "patch"] if args.study == "both" else [args.study]

    for study in studies:
        methods = args.window_methods if study == "window" else args.patch_methods
        settings = iter_settings(study, args.window_values, args.patch_values)
        for setting in settings:
            setting.run_root.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "scripts/run_table56_evals.py",
                "--methods", *methods,
                "--backbones", *args.backbones,
                "--datasets", *args.datasets,
                "--rounds", *[str(round_id) for round_id in args.rounds],
                "--processed-root", str(setting.processed_root),
                "--artifacts-root", str(setting.artifacts_root),
                "--run-root", str(setting.run_root),
                "--temperature", str(args.temperature),
                "--max-tokens", str(args.max_tokens),
            ]
            if args.base_url is not None:
                cmd.extend(["--base-url", args.base_url])
            if args.api_key is not None:
                cmd.extend(["--api-key", args.api_key])
            if args.prop_base_url is not None:
                cmd.extend(["--prop-base-url", args.prop_base_url])
            if args.prop_api_key is not None:
                cmd.extend(["--prop-api-key", args.prop_api_key])
            if args.aggregate_only:
                cmd.append("--aggregate-only")
            run(cmd, cwd=FINALIZED_ROOT)

        run(
            [
                sys.executable,
                "scripts/aggregate_sensitivity_metrics.py",
                "--study", study,
                "--window-values", *[str(value) for value in args.window_values],
                "--patch-values", *[str(value) for value in args.patch_values],
            ],
            cwd=ROOT,
        )

        if not args.skip_plot:
            run(
                [
                    sys.executable,
                    "scripts/plot_sensitivity_results.py",
                    "--study", study,
                ],
                cwd=ROOT,
            )


if __name__ == "__main__":
    main()
