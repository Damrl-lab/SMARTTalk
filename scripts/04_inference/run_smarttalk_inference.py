#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smarttalk.common.config import add_config_argument, load_config
from smarttalk.inference.pipeline import run_smarttalk


def main() -> None:
    parser = add_config_argument(argparse.ArgumentParser(description="Run SMARTTalk online inference."))
    parser.add_argument("--from-cache", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.from_cache:
        print("Using bundled cached outputs; no live inference launched.")
        return
    processed_root = str((ROOT / cfg["processed_root"]).resolve())
    artifacts_root = str((ROOT / cfg["artifacts_root"]).resolve())
    sampled_indices = str((ROOT / "data" / "splits" / "sampled_test_1to23" / "sampled_test_indices.csv").resolve())
    run_smarttalk(
        "--dataset-name", cfg["dataset"],
        "--round", str(cfg.get("round", 1)),
        "--processed-root", processed_root,
        "--artifact-root", artifacts_root,
        "--healthy-per-fail", str(cfg.get("healthy_per_failed", 23)),
        "--sampled-indices-csv", sampled_indices,
        "--model-name", cfg["llm_model_name"],
        "--base-url", cfg["base_url"],
        "--api-key", cfg["api_key"],
    )


if __name__ == "__main__":
    main()
