#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smarttalk.ablation.pipeline import make_ablation_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Render paper-ready ablation figures from cached sensitivity-study curves.")
    parser.parse_args()
    make_ablation_figures()


if __name__ == "__main__":
    main()
