#!/usr/bin/env python3
"""
Aggregate judge scores and perturbation metrics into a paper-shaped Table 7 CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


BACKBONE_ORDER = ["OS1", "OS2", "OS3", "OS4", "PROP"]


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: List[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate_table7(run_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for backbone in BACKBONE_ORDER:
        judge_scores: List[float] = []
        rec_scores: List[float] = []
        attr_sens_hits = attr_sens_total = 0
        act_dir_hits = act_dir_total = 0

        for judge_csv in sorted((run_root / "judge" / backbone).glob("*.csv")):
            with judge_csv.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    judge_scores.append(float(row["exp_score"]))
                    rec_scores.append(float(row["rec_score"]))

        for metrics_json in sorted((run_root / "perturb" / backbone).glob("*_metrics.json")):
            payload = load_json(metrics_json)
            attr_sens_hits += int(payload["attr_sens_hits"])
            attr_sens_total += int(payload["attr_sens_total"])
            act_dir_hits += int(payload["act_dir_hits"])
            act_dir_total += int(payload["act_dir_total"])

        rows.append(
            {
                "backbone": backbone,
                "exp_score": mean(judge_scores),
                "rec_score": mean(rec_scores),
                "attr_sens": (attr_sens_hits / attr_sens_total) if attr_sens_total else None,
                "act_dir_acc": (act_dir_hits / act_dir_total) if act_dir_total else None,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-run Table 7 judge and perturbation outputs.",
    )
    parser.add_argument("--run-root", type=str, required=True,
                        help="Root folder created by scripts/run_table7_pipeline.py.")
    parser.add_argument("--output-csv", type=str, required=True,
                        help="Path to write aggregated Table 7 CSV.")
    args = parser.parse_args()

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = aggregate_table7(Path(args.run_root))
    df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
