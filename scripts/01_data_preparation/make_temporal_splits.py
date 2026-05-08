#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smarttalk.common.config import add_config_argument, load_config
from smarttalk.data.pipeline import run_make_temporal_splits


def main() -> None:
    parser = add_config_argument(argparse.ArgumentParser(description="Build MB1/MB2 train/val/test window splits."))
    args = parser.parse_args()
    cfg = load_config(args.config)
    dataset_by_model_root = str((ROOT / cfg["dataset_by_model_root"]).resolve())
    failure_tag_path = str((ROOT / cfg["failure_tag_path"]).resolve())
    processed_root = str((ROOT / cfg["processed_root"]).resolve())
    run_make_temporal_splits(
        "--dataset-name", cfg["dataset"],
        "--rounds", *[str(v) for v in cfg.get("rounds", [cfg.get("round", 1)])],
        "--dataset-by-model-root", dataset_by_model_root,
        "--failure-tag-path", failure_tag_path,
        "--processed-root", processed_root,
        "--window-size", str(cfg["window_size"]),
        "--fail-horizon-days", str(cfg.get("fail_horizon_days", 30)),
    )


if __name__ == "__main__":
    main()
