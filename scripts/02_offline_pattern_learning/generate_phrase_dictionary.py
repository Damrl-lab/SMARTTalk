#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smarttalk.patterns.pipeline import run_generate_phrase_dictionary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export phrase-dictionary statistics from existing artifacts.")
    parser.add_argument("--artifact-root", type=str, default="artifacts/checkpoints/by_round")
    parser.add_argument("--output-root", type=str, default="results/figures/phrase_dictionary")
    args = parser.parse_args()
    run_generate_phrase_dictionary(
        "--artifact-root", args.artifact_root,
        "--output-root", args.output_root,
    )


if __name__ == "__main__":
    main()
