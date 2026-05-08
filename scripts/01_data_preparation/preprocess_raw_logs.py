#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smarttalk.common.config import add_config_argument, load_config
from smarttalk.data.pipeline import run_preprocess_raw_logs


def main() -> None:
    parser = add_config_argument(argparse.ArgumentParser(description="Filter raw SMART logs into per-model daily CSVs."))
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_preprocess_raw_logs(
        "--dataset-root", cfg.get("raw_source_root", "data/raw/source_logs"),
        "--output-root", cfg.get("dataset_by_model_root", "data/raw/dataset_by_model"),
    )


if __name__ == "__main__":
    main()
