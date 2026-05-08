#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smarttalk.common.config import add_config_argument, load_config
from smarttalk.ablation.pipeline import run_ablation_bundle


def main() -> None:
    parser = add_config_argument(argparse.ArgumentParser(description="Run the L-sensitivity ablation bundle."))
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_ablation_bundle(
        "--datasets", *cfg["datasets"],
        "--rounds", *[str(v) for v in cfg["rounds"]],
        "--window-values", "30",
        "--patch-values", *[str(v) for v in cfg["patch_values"]],
        "--backbones", *cfg["backbones"],
        "--device", cfg.get("device", "cuda:0"),
    )


if __name__ == "__main__":
    main()
