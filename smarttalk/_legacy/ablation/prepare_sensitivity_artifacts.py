#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from sensitivity_common import (
    DEFAULT_DATASETS,
    DEFAULT_PATCH_VALUES,
    DEFAULT_ROUNDS,
    DEFAULT_WINDOW_VALUES,
    FINALIZED_ROOT,
    dataset_by_model_root,
    failure_tag_path,
    iter_settings,
)


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def needs_processed(setting_root: Path, datasets: list[str], rounds: list[int]) -> bool:
    for dataset in datasets:
        for round_id in rounds:
            split_dir = setting_root / f"{dataset}_round{round_id}"
            if not (split_dir / "train.npz").exists():
                return True
            if not (split_dir / "val.npz").exists():
                return True
            if not (split_dir / "test.npz").exists():
                return True
    return False


def needs_artifacts(setting_root: Path, datasets: list[str], rounds: list[int]) -> bool:
    for dataset in datasets:
        prefix = dataset.lower()
        for round_id in rounds:
            artifact_dir = setting_root / f"{dataset}_round{round_id}"
            if not (artifact_dir / f"{prefix}_prototypes_with_phrases.npz").exists():
                return True
            if not (artifact_dir / f"{prefix}_test_prototypes.npz").exists():
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare processed splits and SMARTTalk artifacts for N/L sensitivity studies.",
    )
    parser.add_argument("--study", choices=["window", "patch", "both"], default="both")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=["MB1", "MB2"])
    parser.add_argument("--rounds", nargs="+", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--window-values", nargs="+", type=int, default=DEFAULT_WINDOW_VALUES)
    parser.add_argument("--patch-values", nargs="+", type=int, default=DEFAULT_PATCH_VALUES)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fail-horizon-days", type=int, default=30)
    parser.add_argument("--processed-only", action="store_true",
                        help="Build processed splits only and skip SMARTTalk artifact generation.")
    parser.add_argument("--generate-prototype-figures", action="store_true",
                        help="Also regenerate prototype-visualization figures for each setting/dataset/round.")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild outputs even if the expected files already exist.")
    args = parser.parse_args()

    studies = ["window", "patch"] if args.study == "both" else [args.study]
    dataset_name_arg = "ALL" if sorted(args.datasets) == ["MB1", "MB2"] else None

    for study in studies:
        settings = iter_settings(study, args.window_values, args.patch_values)
        for setting in settings:
            setting.processed_root.mkdir(parents=True, exist_ok=True)
            setting.artifacts_root.mkdir(parents=True, exist_ok=True)

            if args.force or needs_processed(setting.processed_root, args.datasets, args.rounds):
                build_cmd = [
                    sys.executable,
                    "scripts/build_processed_splits.py",
                    "--rounds", *[str(round_id) for round_id in args.rounds],
                    "--dataset-by-model-root", str(dataset_by_model_root()),
                    "--failure-tag-path", str(failure_tag_path()),
                    "--processed-root", str(setting.processed_root),
                    "--window-size", str(setting.window_days),
                    "--fail-horizon-days", str(args.fail_horizon_days),
                ]
                if dataset_name_arg is not None:
                    build_cmd.extend(["--dataset-name", dataset_name_arg])
                else:
                    for dataset in args.datasets:
                        run(build_cmd + ["--dataset-name", dataset], cwd=FINALIZED_ROOT)
                    build_cmd = []
                if build_cmd:
                    run(build_cmd, cwd=FINALIZED_ROOT)
            else:
                print(f"Skipping processed splits for {setting.slug}; files already exist.")

            if args.processed_only:
                continue

            if not args.force and not needs_artifacts(setting.artifacts_root, args.datasets, args.rounds):
                print(f"Skipping SMARTTalk artifacts for {setting.slug}; files already exist.")
                continue

            for dataset in args.datasets:
                for round_id in args.rounds:
                    run(
                        [
                            sys.executable,
                            "scripts/build_offline_artifacts.py",
                            "--dataset-name", dataset,
                            "--round", str(round_id),
                            "--device", args.device,
                            "--processed-root", str(setting.processed_root),
                            "--artifacts-root", str(setting.artifacts_root),
                            "--patch-len", str(setting.patch_len),
                        ],
                        cwd=FINALIZED_ROOT,
                    )

                    if args.generate_prototype_figures:
                        figure_root = setting.results_root / "prototype_figures" / f"{dataset}_round{round_id}"
                        run(
                            [
                                sys.executable,
                                "scripts/generate_prototype_figures.py",
                                "--dataset-name", dataset,
                                "--round", str(round_id),
                                "--device", args.device,
                                "--processed-root", str(setting.processed_root),
                                "--artifacts-root", str(setting.artifacts_root),
                                "--output-root", str(figure_root),
                            ],
                            cwd=FINALIZED_ROOT,
                        )


if __name__ == "__main__":
    main()
