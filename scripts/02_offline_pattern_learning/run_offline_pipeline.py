#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smarttalk.common.config import add_config_argument, load_config
from smarttalk.patterns.pipeline import run_offline_pipeline


def main() -> None:
    parser = add_config_argument(argparse.ArgumentParser(description="Run SMARTTalk offline pattern learning for one dataset round."))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--generate-figures", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    processed_root = str((ROOT / cfg["processed_root"]).resolve())
    artifacts_root = str((ROOT / cfg["artifacts_root"]).resolve())
    cmd = [
        "--dataset-name", cfg["dataset"],
        "--round", str(cfg.get("round", 1)),
        "--device", args.device or cfg.get("device", "cpu"),
        "--processed-root", processed_root,
        "--artifacts-root", artifacts_root,
        "--patch-len", str(cfg.get("patch_length", 5)),
    ]
    if args.generate_figures:
        cmd.append("--generate-figures")
    run_offline_pipeline(*cmd)


if __name__ == "__main__":
    main()
