#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from sensitivity_common import (
    DEFAULT_BACKBONES,
    DEFAULT_DATASETS,
    DEFAULT_PATCH_VALUES,
    DEFAULT_ROUNDS,
    DEFAULT_WINDOW_VALUES,
    ROOT,
)


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full rebuttal-facing N/L sensitivity pipeline from artifact preparation through plotting.",
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=["MB1", "MB2"])
    parser.add_argument("--rounds", nargs="+", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--window-values", nargs="+", type=int, default=DEFAULT_WINDOW_VALUES)
    parser.add_argument("--patch-values", nargs="+", type=int, default=DEFAULT_PATCH_VALUES)
    parser.add_argument("--backbones", nargs="+", default=DEFAULT_BACKBONES,
                        help="LLM backbones for rebuttal runs. Defaults to OS3 + PROP.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Build processed splits, SMARTTalk artifacts, and prototype figures only.")
    parser.add_argument("--skip-prototype-figures", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--prop-base-url", type=str, default=None)
    parser.add_argument("--prop-api-key", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    prepare_base = [
        sys.executable,
        "scripts/prepare_sensitivity_artifacts.py",
        "--datasets", *args.datasets,
        "--rounds", *[str(value) for value in args.rounds],
        "--window-values", *[str(value) for value in args.window_values],
        "--patch-values", *[str(value) for value in args.patch_values],
        "--device", args.device,
    ]
    if args.force:
        prepare_base.append("--force")
    if not args.skip_prototype_figures:
        prepare_base.append("--generate-prototype-figures")

    run(prepare_base + ["--study", "window"], cwd=ROOT)
    run(prepare_base + ["--study", "patch"], cwd=ROOT)

    if not args.prepare_only:
        eval_base = [
            sys.executable,
            "scripts/run_sensitivity_study.py",
            "--datasets", *args.datasets,
            "--rounds", *[str(value) for value in args.rounds],
            "--window-values", *[str(value) for value in args.window_values],
            "--patch-values", *[str(value) for value in args.patch_values],
            "--backbones", *args.backbones,
            "--temperature", str(args.temperature),
            "--max-tokens", str(args.max_tokens),
            "--skip-plot",
        ]
        if args.base_url is not None:
            eval_base.extend(["--base-url", args.base_url])
        if args.api_key is not None:
            eval_base.extend(["--api-key", args.api_key])
        if args.prop_base_url is not None:
            eval_base.extend(["--prop-base-url", args.prop_base_url])
        if args.prop_api_key is not None:
            eval_base.extend(["--prop-api-key", args.prop_api_key])

        run(eval_base + ["--study", "window"], cwd=ROOT)
        run(eval_base + ["--study", "patch"], cwd=ROOT)
        run([sys.executable, "scripts/plot_sensitivity_results.py", "--study", "both"], cwd=ROOT)

    run([sys.executable, "scripts/ablation_readable_figures.py"], cwd=ROOT)


if __name__ == "__main__":
    main()
