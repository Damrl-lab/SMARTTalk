#!/usr/bin/env python3
"""
Aggregate per-round LLM evaluation outputs into paper-shaped Table 5 and Table 6 CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


METHOD_ORDER = ["Raw-LLM", "Heuristic-LLM", "SMARTTalk"]
METHOD_SLUGS = {
    "Raw-LLM": "raw",
    "Heuristic-LLM": "heuristic",
    "SMARTTalk": "smarttalk",
}
BACKBONE_ORDER = ["OS1", "OS2", "OS3", "OS4", "PROP"]
TTF_BUCKETS = ["<7", "7-30", ">30"]
TTF_BUCKET_MIDPOINTS = {"<7": 3.5, "7-30": 18.5, ">30": 45.0}


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def f05_score(precision: float, recall: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = 0.25
    return (1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def ttf_to_bucket(ttf_days: float) -> str:
    if ttf_days < 7:
        return "<7"
    if ttf_days <= 30:
        return "7-30"
    return ">30"


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_status(run_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for method_name in METHOD_ORDER:
        slug = METHOD_SLUGS[method_name]
        for backbone in BACKBONE_ORDER:
            result: Dict[str, object] = {"method": method_name, "backbone": backbone}
            for dataset in ["MB1", "MB2"]:
                tp = fp = tn = fn = 0
                metric_paths = sorted((run_root / slug / backbone).glob(f"{dataset}_round*_metrics.json"))
                for metrics_path in metric_paths:
                    payload = load_json(metrics_path)["classification"]
                    tp += int(payload["tp"])
                    fp += int(payload["fp"])
                    tn += int(payload["tn"])
                    fn += int(payload["fn"])
                if not metric_paths:
                    result[f"{dataset.lower()}_precision"] = None
                    result[f"{dataset.lower()}_recall"] = None
                    result[f"{dataset.lower()}_f05"] = None
                    result[f"{dataset.lower()}_fpr"] = None
                    result[f"{dataset.lower()}_fnr"] = None
                    continue
                precision = safe_div(tp, tp + fp)
                recall = safe_div(tp, tp + fn)
                result[f"{dataset.lower()}_precision"] = precision
                result[f"{dataset.lower()}_recall"] = recall
                result[f"{dataset.lower()}_f05"] = f05_score(precision, recall)
                result[f"{dataset.lower()}_fpr"] = safe_div(fp, fp + tn)
                result[f"{dataset.lower()}_fnr"] = safe_div(fn, fn + tp)
            rows.append(result)
    return pd.DataFrame(rows)


def aggregate_ttf(run_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for backbone in BACKBONE_ORDER:
        result: Dict[str, object] = {"backbone": backbone}
        for dataset in ["MB1", "MB2"]:
            all_rows: List[Dict[str, str]] = []
            for csv_path in sorted((run_root / "smarttalk" / backbone).glob(f"{dataset}_round*_tp.csv")):
                with csv_path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    all_rows.extend(list(reader))

            usable = [
                row for row in all_rows
                if row["ttf_bucket_pred"] in TTF_BUCKET_MIDPOINTS
                and row["ttf_true"] not in {"", "None", "nan"}
            ]
            if not usable:
                result[f"{dataset.lower()}_ttf_f1"] = None
                result[f"{dataset.lower()}_bmae"] = None
                result[f"{dataset.lower()}_cov_pm5"] = None
                continue

            true_ttf = [float(row["ttf_true"]) for row in usable]
            true_bucket = [ttf_to_bucket(value) for value in true_ttf]
            pred_bucket = [row["ttf_bucket_pred"] for row in usable]

            per_bucket_f1 = []
            for bucket in TTF_BUCKETS:
                tp_b = sum(1 for t, p in zip(true_bucket, pred_bucket) if t == bucket and p == bucket)
                fp_b = sum(1 for t, p in zip(true_bucket, pred_bucket) if t != bucket and p == bucket)
                fn_b = sum(1 for t, p in zip(true_bucket, pred_bucket) if t == bucket and p != bucket)
                if tp_b + fn_b == 0:
                    continue
                precision = safe_div(tp_b, tp_b + fp_b)
                recall = safe_div(tp_b, tp_b + fn_b)
                bucket_f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
                per_bucket_f1.append(bucket_f1)

            pred_mid = [TTF_BUCKET_MIDPOINTS[bucket] for bucket in pred_bucket]
            abs_err = [abs(t - p) for t, p in zip(true_ttf, pred_mid)]

            result[f"{dataset.lower()}_ttf_f1"] = sum(per_bucket_f1) / len(per_bucket_f1) if per_bucket_f1 else None
            result[f"{dataset.lower()}_bmae"] = sum(abs_err) / len(abs_err)
            result[f"{dataset.lower()}_cov_pm5"] = sum(1 for err in abs_err if err <= 5.0) / len(abs_err)
        rows.append(result)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-round SMARTTalk LLM outputs into Table 5 and Table 6 CSVs.",
    )
    parser.add_argument("--run-root", type=str, required=True,
                        help="Root folder created by scripts/run_table56_evals.py.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Folder where aggregated table CSVs are written.")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table5 = aggregate_status(run_root)
    table6 = aggregate_ttf(run_root)

    table5.to_csv(output_dir / "table5_status_from_runs.csv", index=False)
    table6.to_csv(output_dir / "table6_ttf_from_runs.csv", index=False)
    print(f"Wrote {output_dir / 'table5_status_from_runs.csv'}")
    print(f"Wrote {output_dir / 'table6_ttf_from_runs.csv'}")


if __name__ == "__main__":
    main()
