#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smarttalk.common.config import add_config_argument, load_config
from smarttalk.data.schema import summarize_npz_split


def main() -> None:
    parser = add_config_argument(argparse.ArgumentParser(description="Validate a processed split schema."))
    parser.add_argument("--split", type=str, default=None, help="Optional explicit .npz path.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.split:
        split_path = Path(args.split)
    else:
        dataset = cfg["dataset"]
        round_id = cfg.get("round", 1)
        cached_test_split = ROOT / "data" / "splits" / f"{dataset}_round{round_id}" / "test.npz"
        if cached_test_split.exists():
            split_path = cached_test_split
        else:
            split_path = ROOT / "data" / "sample_data" / f"{dataset}_round1_test_sample.npz"
    print(summarize_npz_split(split_path))


if __name__ == "__main__":
    main()
