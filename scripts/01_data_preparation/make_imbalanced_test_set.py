#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smarttalk.common.config import add_config_argument, load_config
from smarttalk.data.pipeline import run_make_sampled_test


def main() -> None:
    parser = add_config_argument(argparse.ArgumentParser(description="Create the fixed 1:23 sampled test set."))
    args = parser.parse_args()
    cfg = load_config(args.config)
    processed_root = str((ROOT / cfg["processed_root"]).resolve())
    output_dir = str((ROOT / "data" / "splits" / "sampled_test_1to23").resolve())
    run_make_sampled_test(
        "--processed-root", processed_root,
        "--output-dir", output_dir,
        "--healthy-per-failed", str(cfg.get("healthy_per_failed", 23)),
        "--seed", str(cfg.get("seed", 2026)),
    )


if __name__ == "__main__":
    main()
