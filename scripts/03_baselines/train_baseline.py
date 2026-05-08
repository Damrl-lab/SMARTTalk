#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smarttalk.common.config import add_config_argument, load_config
from smarttalk.baselines.pipeline import run_baseline


def main() -> None:
    parser = add_config_argument(argparse.ArgumentParser(description="Run one numerical baseline."))
    parser.add_argument("--model", choices=["rf", "nn", "ec", "ae", "lstm", "mvtrf", "msfrd"], required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_dir = str((ROOT / cfg["processed_root"] / f"{cfg['dataset']}_round{cfg.get('round', 1)}").resolve())
    run_baseline(args.model, "--data_dir", data_dir)


if __name__ == "__main__":
    main()
