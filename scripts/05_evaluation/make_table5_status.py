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
from smarttalk.evaluation.pipeline import make_table5_status


def main() -> None:
    parser = add_config_argument(argparse.ArgumentParser(description="Regenerate Table 5 status outputs with sampled-set FPR/FNR."))
    args = parser.parse_args()
    cfg = load_config(args.config)
    processed_root = str((ROOT / cfg["processed_root"]).resolve())
    output_dir = str((ROOT / "data" / "splits" / "sampled_test_1to23").resolve())
    sampled_summary = Path(output_dir) / "sampling_summary.csv"
    mb1_test = Path(processed_root) / "MB1_round1" / "test.npz"
    mb2_test = Path(processed_root) / "MB2_round1" / "test.npz"
    if not sampled_summary.exists():
        if mb1_test.exists() and mb2_test.exists():
            run_make_sampled_test(
                "--processed-root", processed_root,
                "--output-dir", output_dir,
                "--healthy-per-failed", str(cfg.get("healthy_per_failed", 23)),
                "--seed", str(cfg.get("seed", 2026)),
            )
        else:
            raise FileNotFoundError(
                "sampling_summary.csv is missing and the full processed splits are not bundled. "
                "Provide full processed data under data/splits/ or copy the cached sampled-test CSVs."
            )
    make_table5_status()


if __name__ == "__main__":
    main()
