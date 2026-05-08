#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smarttalk.common.config import add_config_argument
from smarttalk.evaluation.pipeline import make_paper_tables


def main() -> None:
    parser = add_config_argument(argparse.ArgumentParser(description="Refresh deterministic Table 6 outputs."))
    parser.parse_args()
    make_paper_tables()


if __name__ == "__main__":
    main()
