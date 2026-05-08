#!/usr/bin/env python3
"""
Rebuild the Figure 3 offline SMARTTalk pipeline for one dataset round.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SMARTTalk offline artifact generation for one dataset round.",
    )
    parser.add_argument("--dataset-name", type=str, default="MB2",
                        choices=["MB1", "MB2"],
                        help="Dataset/model name.")
    parser.add_argument("--round", type=int, default=1,
                        help="Temporal round to rebuild.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device for step1 and phrase generation.")
    parser.add_argument("--processed-root", type=str, default="data/processed",
                        help="Root containing processed train/val/test splits.")
    parser.add_argument("--artifacts-root", type=str, default="data/artifacts",
                        help="Root containing saved SMARTTalk artifacts.")
    parser.add_argument("--patch-len", type=int, default=None,
                        help="Shared patch length for attribute and cross-attribute encoders.")
    parser.add_argument("--patch-len-attr", type=int, default=None,
                        help="Attribute patch length in days. Overrides --patch-len when set.")
    parser.add_argument("--patch-len-cross", type=int, default=None,
                        help="Cross-attribute patch length in days. Overrides --patch-len when set.")
    parser.add_argument("--generate-figures", action="store_true",
                        help="Also regenerate Figure 4 and Figure 5 prototype visualizations.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    artifact_dir = Path(args.artifacts_root) / f"{args.dataset_name}_round{args.round}"
    train_path = Path(args.processed_root) / f"{args.dataset_name}_round{args.round}" / "train.npz"
    patch_len_attr = args.patch_len_attr or args.patch_len or 5
    patch_len_cross = args.patch_len_cross or args.patch_len or 5

    run(
        [
            sys.executable,
            "code/core/step1.py",
            "--mode", "step1",
            "--dataset-name", args.dataset_name,
            "--round", str(args.round),
            "--device", args.device,
            "--processed-root", args.processed_root,
            "--artifacts-root", args.artifacts_root,
            "--patch-len-attr", str(patch_len_attr),
            "--patch-len-cross", str(patch_len_cross),
        ],
        cwd=root,
    )

    for split in ["train", "val", "test"]:
        run(
            [
                sys.executable,
                "code/core/step1.py",
                "--mode", "step2",
                "--split", split,
                "--dataset-name", args.dataset_name,
                "--round", str(args.round),
                "--device", args.device,
                "--processed-root", args.processed_root,
                "--artifacts-root", args.artifacts_root,
                "--patch-len-attr", str(patch_len_attr),
                "--patch-len-cross", str(patch_len_cross),
            ],
            cwd=root,
        )

    run(
        [
            sys.executable,
            "code/core/learn_vocab_from_prototypes.py",
            "--device", args.device,
            "--train-path", str(train_path),
            "--artifact-root", str(artifact_dir),
            "--artifact-prefix", args.dataset_name.lower(),
        ],
        cwd=root,
    )

    if args.generate_figures:
        run(
            [
                sys.executable,
                "scripts/generate_prototype_figures.py",
                "--dataset-name", args.dataset_name,
                "--round", str(args.round),
                "--device", args.device,
                "--processed-root", args.processed_root,
                "--artifacts-root", args.artifacts_root,
                "--output-root", "results/paper_figures",
            ],
            cwd=root,
        )


if __name__ == "__main__":
    main()
