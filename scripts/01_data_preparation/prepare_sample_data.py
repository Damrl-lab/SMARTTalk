#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smarttalk.data.schema import summarize_npz_split
from smarttalk.common.logging_utils import log_step


def main() -> None:
    sample_dir = ROOT / "data" / "sample_data"
    for npz_path in sorted(sample_dir.glob("*.npz")):
        summary = summarize_npz_split(npz_path)
        log_step(f"{npz_path.name}: {summary}")


if __name__ == "__main__":
    main()
