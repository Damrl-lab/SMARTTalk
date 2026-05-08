#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parent
CORE_ROOT = SCRIPT_ROOT.parent / "code" / "core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from sampled_test_utils import DEFAULT_HEALTHY_PER_FAILED, DEFAULT_SAMPLE_SEED, write_sampled_test_tables
from status_sampled_utils import derive_sampled_status_table, format_wide_status_table_for_csv
from status_sampled_utils import load_sampled_supports


ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = ROOT / "configs" / "paper_tables"
OUTPUT_ROOT = ROOT / "results" / "paper_tables"
FIGURE_SOURCE_ROOT = ROOT / "data" / "figures"
FIGURE_OUTPUT_ROOT = ROOT / "results" / "paper_figures"
PROCESSED_ROOT = ROOT / "data" / "processed"
SAMPLED_TEST_ROOT = PROCESSED_ROOT / "sampled_test_1to23"
TABLE5_WITH_RATES_CONFIG = CONFIG_ROOT / "table5_status_with_fpr_fnr.csv"


def normalize_backbone(value: object) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "-"
    return text


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame]:
    table5 = pd.read_csv(CONFIG_ROOT / "table5_status.csv")
    table5_with_rates = (
        pd.read_csv(TABLE5_WITH_RATES_CONFIG)
        if TABLE5_WITH_RATES_CONFIG.exists()
        else None
    )
    table6 = pd.read_csv(CONFIG_ROOT / "table6_ttf.csv")
    table7 = pd.read_csv(CONFIG_ROOT / "table7_explanations.csv")
    return table5, table5_with_rates, table6, table7


def write_csv_copies(
    table5: pd.DataFrame,
    table5_with_rates: pd.DataFrame | None,
    table6: pd.DataFrame,
    table7: pd.DataFrame,
) -> None:
    table5.to_csv(OUTPUT_ROOT / "table5_status.csv", index=False)
    if table5_with_rates is not None:
        shutil.copy2(TABLE5_WITH_RATES_CONFIG, OUTPUT_ROOT / "table5_status_with_fpr_fnr.csv")
    table6.to_csv(OUTPUT_ROOT / "table6_ttf.csv", index=False)
    table7.to_csv(OUTPUT_ROOT / "table7_explanations.csv", index=False)


def write_table5_with_rates_artifacts(
    table5: pd.DataFrame,
    curated_table5_with_rates: pd.DataFrame | None,
) -> None:
    sampled_indices_csv, sampling_summary_csv = write_sampled_test_tables(
        processed_root=PROCESSED_ROOT,
        output_dir=SAMPLED_TEST_ROOT,
        healthy_per_failed=DEFAULT_HEALTHY_PER_FAILED,
        seed=DEFAULT_SAMPLE_SEED,
    )
    sampled_supports = load_sampled_supports(sampling_summary_csv)
    derived_table5_with_rates, detail_rows = derive_sampled_status_table(
        table5,
        sampled_supports,
        sampled_indices_csv=sampled_indices_csv,
        run_root=None,
    )
    final_table5_with_rates = (
        curated_table5_with_rates.copy()
        if curated_table5_with_rates is not None
        else derived_table5_with_rates
    )
    if curated_table5_with_rates is None:
        format_wide_status_table_for_csv(final_table5_with_rates).to_csv(
            OUTPUT_ROOT / "table5_status_with_fpr_fnr.csv",
            index=False,
        )
    detail_rows.to_csv(OUTPUT_ROOT / "table5_status_with_fpr_fnr_details.csv", index=False)
    (OUTPUT_ROOT / "status_test_supports.json").write_text(
        pd.DataFrame(
            [
                {
                    "dataset": dataset,
                    "rounds": ",".join(str(r) for r in support.rounds),
                    "failed": support.positives,
                    "healthy": support.negatives,
                    "healthy_per_failed": support.healthy_per_failed,
                    "seed": support.seed,
                }
                for dataset, support in sampled_supports.items()
            ]
        ).to_json(orient="records", indent=2)
        + "\n"
    )
    (OUTPUT_ROOT / "table5_status_with_fpr_fnr_audit.json").write_text(
        pd.DataFrame(
            [
                {
                    "sampled_indices_csv": str(sampled_indices_csv),
                    "sampling_summary_csv": str(sampling_summary_csv),
                    "healthy_per_failed": DEFAULT_HEALTHY_PER_FAILED,
                    "seed": DEFAULT_SAMPLE_SEED,
                    "paper_table_source": (
                        str(TABLE5_WITH_RATES_CONFIG)
                        if curated_table5_with_rates is not None
                        else "derived_from_sampled_supports"
                    ),
                }
            ]
        ).to_json(orient="records", indent=2)
        + "\n"
    )


def write_summary_markdown(table5: pd.DataFrame, table6: pd.DataFrame, table7: pd.DataFrame) -> None:
    best_mb1 = table5.loc[table5["mb1_f05"].idxmax()]
    best_mb2 = table5.loc[table5["mb2_f05"].idxmax()]
    best_ttf_mb1 = table6.loc[table6["mb1_ttf_f1"].idxmax()]
    best_ttf_mb2 = table6.loc[table6["mb2_ttf_f1"].idxmax()]
    best_exp = table7.loc[table7["exp_score"].idxmax()]
    best_rec = table7.loc[table7["rec_score"].idxmax()]
    best_mb1_name = best_mb1["method"]
    if normalize_backbone(best_mb1["backbone"]) != "-":
        best_mb1_name = f"{best_mb1_name} ({normalize_backbone(best_mb1['backbone'])})"
    best_mb2_name = best_mb2["method"]
    if normalize_backbone(best_mb2["backbone"]) != "-":
        best_mb2_name = f"{best_mb2_name} ({normalize_backbone(best_mb2['backbone'])})"

    lines = [
        "# Paper Results Summary",
        "",
        "These files are the exact paper-level values transcribed from the SMARTTalk OSDI'26 manuscript so the package can regenerate the reported tables without using rounded-value reconstruction helpers.",
        "",
        "## Best Status F0.5",
        f"- MB1: `{best_mb1_name}` with F0.5 = {best_mb1['mb1_f05']:.2f}.",
        f"- MB2: `{best_mb2_name}` with F0.5 = {best_mb2['mb2_f05']:.2f}.",
        "",
        "## Best TTF Macro-F1",
        f"- MB1: `{best_ttf_mb1['backbone']}` with TTF F1 = {best_ttf_mb1['mb1_ttf_f1']:.2f}.",
        f"- MB2: `{best_ttf_mb2['backbone']}` with TTF F1 = {best_ttf_mb2['mb2_ttf_f1']:.2f}.",
        "",
        "## Best Explanation / Recommendation Scores",
        f"- Highest explanation score: `{best_exp['backbone']}` at {best_exp['exp_score']:.2f}.",
        f"- Highest recommendation score: `{best_rec['backbone']}` at {best_rec['rec_score']:.2f}.",
    ]
    (OUTPUT_ROOT / "paper_results_summary.md").write_text("\n".join(lines) + "\n")


def copy_paper_figures() -> None:
    FIGURE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for source in sorted(FIGURE_SOURCE_ROOT.glob("*")):
        if source.is_file():
            shutil.copy2(source, FIGURE_OUTPUT_ROOT / source.name)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    table5, table5_with_rates, table6, table7 = load_tables()
    write_csv_copies(table5, table5_with_rates, table6, table7)
    write_table5_with_rates_artifacts(table5, table5_with_rates)
    write_summary_markdown(table5, table6, table7)
    copy_paper_figures()

    print(f"Wrote {OUTPUT_ROOT / 'table5_status.csv'}")
    print(f"Wrote {OUTPUT_ROOT / 'table5_status_with_fpr_fnr.csv'}")
    print(f"Wrote {OUTPUT_ROOT / 'table5_status_with_fpr_fnr_details.csv'}")
    print(f"Wrote {OUTPUT_ROOT / 'status_test_supports.json'}")
    print(f"Wrote {OUTPUT_ROOT / 'table5_status_with_fpr_fnr_audit.json'}")
    print(f"Wrote {OUTPUT_ROOT / 'table6_ttf.csv'}")
    print(f"Wrote {OUTPUT_ROOT / 'table7_explanations.csv'}")
    print(f"Wrote {OUTPUT_ROOT / 'paper_results_summary.md'}")
    print(f"Copied paper figures to {FIGURE_OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
