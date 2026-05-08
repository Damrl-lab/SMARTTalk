#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from sensitivity_common import (
    DEFAULT_PATCH_VALUES,
    DEFAULT_WINDOW_VALUES,
    ROOT,
    iter_settings,
)


DATASET_COLUMNS = {
    "MB1": ("mb1_precision", "mb1_recall", "mb1_f05", "mb1_fpr", "mb1_fnr"),
    "MB2": ("mb2_precision", "mb2_recall", "mb2_f05", "mb2_fpr", "mb2_fnr"),
}


def load_setting_rows(setting) -> List[Dict[str, object]]:
    table_path = setting.run_root / "aggregated" / "table5_status_from_runs.csv"
    if not table_path.exists():
        print(f"Skipping {setting.slug}; missing {table_path}")
        return []

    df = pd.read_csv(table_path)
    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        method = str(row["method"])
        backbone = str(row["backbone"])
        for dataset, columns in DATASET_COLUMNS.items():
            precision_col, recall_col, f05_col, fpr_col, fnr_col = columns
            precision = row.get(precision_col)
            recall = row.get(recall_col)
            f05 = row.get(f05_col)
            fpr = row.get(fpr_col)
            fnr = row.get(fnr_col)
            if pd.isna(precision) or pd.isna(recall) or pd.isna(f05):
                continue
            rows.append(
                {
                    "study": setting.study,
                    "study_folder": setting.study_folder,
                    "setting_slug": setting.slug,
                    "window_days": setting.window_days,
                    "patch_len": setting.patch_len,
                    "dataset": dataset,
                    "method": method,
                    "backbone": backbone,
                    "precision": float(precision),
                    "recall": float(recall),
                    "f05": float(f05),
                    "fpr": float(fpr),
                    "fnr": float(fnr),
                    "is_baseline": bool(setting.window_days == 30 and setting.patch_len == 5),
                }
            )
    return rows


def aggregate_study(study: str, window_values: list[int], patch_values: list[int]) -> None:
    rows: List[Dict[str, object]] = []
    settings = iter_settings(study, window_values, patch_values)
    for setting in settings:
        rows.extend(load_setting_rows(setting))

    if not rows:
        print(f"No aggregated status rows found for study={study}.")
        return

    output_root = ROOT / "results" / ("window_sensitivity" if study == "window" else "patch_sensitivity")
    output_root.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows).sort_values(
        by=["dataset", "method", "backbone", "window_days", "patch_len"],
        kind="stable",
    )
    df.to_csv(output_root / "status_sensitivity_long.csv", index=False)

    mean_df = (
        df.groupby(["study", "dataset", "method", "backbone"], as_index=False)[["precision", "recall", "f05", "fpr", "fnr"]]
        .mean()
    )
    mean_df.to_csv(output_root / "status_sensitivity_mean_by_line.csv", index=False)

    print(f"Wrote {output_root / 'status_sensitivity_long.csv'}")
    print(f"Wrote {output_root / 'status_sensitivity_mean_by_line.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate N/L sensitivity Table 5 outputs into long-form CSVs.",
    )
    parser.add_argument("--study", choices=["window", "patch", "both"], default="both")
    parser.add_argument("--window-values", nargs="+", type=int, default=DEFAULT_WINDOW_VALUES)
    parser.add_argument("--patch-values", nargs="+", type=int, default=DEFAULT_PATCH_VALUES)
    args = parser.parse_args()

    studies = ["window", "patch"] if args.study == "both" else [args.study]
    for study in studies:
        aggregate_study(study, args.window_values, args.patch_values)


if __name__ == "__main__":
    main()
