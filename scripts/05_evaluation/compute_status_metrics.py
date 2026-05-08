#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smarttalk.evaluation.status_metrics import compute_status_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute precision/recall/F0.5/FPR/FNR from y_true and y_pred arrays in JSON.")
    parser.add_argument("--json", required=True, help="JSON file with y_true and y_pred lists.")
    args = parser.parse_args()
    payload = json.loads(Path(args.json).read_text(encoding="utf-8"))
    metrics = compute_status_metrics(payload["y_true"], payload["y_pred"])
    print(metrics)


if __name__ == "__main__":
    main()
