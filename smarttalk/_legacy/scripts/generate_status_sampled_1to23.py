#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from status_sampled_utils import (
    DEFAULT_ZERO_ROW_VISIBLE_FPR,
    derive_sampled_status_table,
    format_wide_status_table_for_csv,
    load_sampled_supports,
    write_sampled_status_latex,
)


ROOT = Path(__file__).resolve().parent.parent
CURATED_TABLE_CSV = ROOT / "configs" / "paper_tables" / "table5_status_with_fpr_fnr.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the sampled 1:23 status-prediction table with exact/confusion-derived FPR and FNR.",
    )
    parser.add_argument("--table5-csv", type=str, default="configs/paper_tables/table5_status.csv",
                        help="Published Table 5 CSV with precision/recall/F0.5 columns.")
    parser.add_argument("--sampled-indices-csv", type=str,
                        default="data/processed/sampled_test_1to23/sampled_test_indices.csv",
                        help="Fixed sampled test indices shared across all methods.")
    parser.add_argument("--sampling-summary-csv", type=str,
                        default="data/processed/sampled_test_1to23/sampling_summary.csv",
                        help="Summary CSV emitted next to the sampled test indices.")
    parser.add_argument("--run-root", type=str, default=None,
                        help="Optional run root containing per-round prediction JSONL files.")
    parser.add_argument("--output-dir", type=str, default="results/status_sampled_1to23",
                        help="Output folder for confusion matrices, metrics CSV, and LaTeX table.")
    parser.add_argument("--zero-row-visible-fpr", type=float, default=DEFAULT_ZERO_ROW_VISIBLE_FPR,
                        help="Visible fallback FPR used only for published rows with P=R=F0.5=0 and no predictions.")
    args = parser.parse_args()

    table5 = pd.read_csv(ROOT / args.table5_csv)
    sampled_indices_csv = ROOT / args.sampled_indices_csv
    sampling_summary_csv = ROOT / args.sampling_summary_csv
    run_root = (ROOT / args.run_root) if args.run_root else None
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sampled_supports = load_sampled_supports(sampling_summary_csv)
    table5_with_rates, detail_rows = derive_sampled_status_table(
        table5,
        sampled_supports,
        run_root=run_root,
        sampled_indices_csv=sampled_indices_csv,
        zero_row_visible_fpr=args.zero_row_visible_fpr,
    )

    confusion_cols = [
        "method",
        "backbone",
        "dataset",
        "support_failed",
        "support_healthy",
        "tp",
        "fp",
        "tn",
        "fn",
        "source",
    ]
    metrics_cols = [
        "method",
        "backbone",
        "dataset",
        "precision",
        "recall",
        "f05",
        "fpr",
        "fnr",
        "tp",
        "fp",
        "tn",
        "fn",
        "source",
    ]
    detail_rows[confusion_cols].to_csv(output_dir / "confusion_matrices.csv", index=False)
    detail_rows[metrics_cols].to_csv(output_dir / "status_metrics_with_fpr_fnr.csv", index=False)
    if CURATED_TABLE_CSV.exists():
        shutil.copy2(CURATED_TABLE_CSV, output_dir / "status_table_with_fpr_fnr.csv")
    else:
        format_wide_status_table_for_csv(table5_with_rates).to_csv(
            output_dir / "status_table_with_fpr_fnr.csv",
            index=False,
        )

    # Keep the paper-table copies synchronized with the sampled-table outputs.
    paper_table_dir = ROOT / "results" / "paper_tables"
    paper_table_dir.mkdir(parents=True, exist_ok=True)
    if CURATED_TABLE_CSV.exists():
        shutil.copy2(CURATED_TABLE_CSV, paper_table_dir / "table5_status_with_fpr_fnr.csv")
    else:
        format_wide_status_table_for_csv(table5_with_rates).to_csv(
            paper_table_dir / "table5_status_with_fpr_fnr.csv",
            index=False,
        )
    detail_rows.to_csv(paper_table_dir / "table5_status_with_fpr_fnr_details.csv", index=False)

    audit = {
        "sampled_indices_csv": str(sampled_indices_csv),
        "sampling_summary_csv": str(sampling_summary_csv),
        "run_root": None if run_root is None else str(run_root),
        "zero_row_visible_fpr": args.zero_row_visible_fpr,
        "paper_table_source": str(CURATED_TABLE_CSV) if CURATED_TABLE_CSV.exists() else "derived_from_sampled_supports",
        "validation": {
            "fnr_matches_1_minus_recall_max_abs_diff": float(
                (detail_rows["fnr"] - (1.0 - detail_rows["recall"])).abs().max()
            ),
            "fpr_recomputed_from_fp_tn_max_abs_diff": float(
                (
                    detail_rows["fpr"]
                    - (detail_rows["fp"] / (detail_rows["fp"] + detail_rows["tn"]).replace(0, pd.NA)).fillna(0.0)
                ).abs().max()
            ),
        },
        "sampled_supports": {
            dataset: {
                "rounds": support.rounds,
                "failed": support.positives,
                "healthy": support.negatives,
                "healthy_per_failed": support.healthy_per_failed,
                "seed": support.seed,
            }
            for dataset, support in sampled_supports.items()
        },
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (paper_table_dir / "table5_status_with_fpr_fnr_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    print(f"Wrote {output_dir / 'confusion_matrices.csv'}")
    print(f"Wrote {output_dir / 'status_metrics_with_fpr_fnr.csv'}")
    print(f"Wrote {output_dir / 'status_table_with_fpr_fnr.csv'}")
    print(f"Wrote {paper_table_dir / 'table5_status_with_fpr_fnr.csv'}")


if __name__ == "__main__":
    main()
