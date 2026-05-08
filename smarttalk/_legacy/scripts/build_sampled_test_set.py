#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CORE_ROOT = ROOT / "code" / "core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from sampled_test_utils import (  # noqa: E402
    DEFAULT_HEALTHY_PER_FAILED,
    DEFAULT_SAMPLE_SEED,
    write_sampled_test_tables,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one fixed sampled test set shared by all status-prediction methods.",
    )
    parser.add_argument("--processed-root", type=str, default="data/processed",
                        help="Root containing MB1_round*/test.npz and MB2_round*/test.npz.")
    parser.add_argument("--output-dir", type=str, default="data/processed/sampled_test_1to23",
                        help="Output folder for sampled_test_indices.csv and sampling_summary.csv.")
    parser.add_argument("--healthy-per-failed", type=float, default=DEFAULT_HEALTHY_PER_FAILED,
                        help="Number of healthy windows sampled per failed window.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED,
                        help="Random seed used for healthy-window sampling.")
    args = parser.parse_args()

    processed_root = ROOT / args.processed_root
    output_dir = ROOT / args.output_dir

    indices_csv, summary_csv = write_sampled_test_tables(
        processed_root=processed_root,
        output_dir=output_dir,
        healthy_per_failed=args.healthy_per_failed,
        seed=args.seed,
    )

    print(f"Wrote {indices_csv}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
