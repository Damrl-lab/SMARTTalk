#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smarttalk.evaluation.pipeline import make_paper_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh cached paper tables after running baseline code.")
    parser.parse_args()
    make_paper_tables()


if __name__ == "__main__":
    main()
