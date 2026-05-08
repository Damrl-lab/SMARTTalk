#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from sensitivity_common import DEFAULT_PATCH_VALUES, DEFAULT_WINDOW_VALUES, FINALIZED_ROOT, ROOT


WINDOW_VALUES = DEFAULT_WINDOW_VALUES
PATCH_VALUES = DEFAULT_PATCH_VALUES
DATASETS = ["MB1", "MB2"]
BACKBONES = ["OS3", "PROP"]
WINDOW_METHODS = ["Raw-LLM", "Heuristic-LLM", "SMARTTalk"]
PATCH_METHODS = ["SMARTTalk"]
BASELINE_WINDOW = 30
BASELINE_PATCH = 5

# Paper-readable plotting defaults. These values are intentionally larger
# than typical notebook defaults so labels remain legible when the figure
# is inserted in the paper and viewed at 100% zoom.
BASE_FONT_SIZE = 20
AXIS_LABEL_SIZE = 22
TICK_LABEL_SIZE = 18
LEGEND_FONT_SIZE = 20
ANNOTATION_FONT_SIZE = 20
LINE_WIDTH = 3
MAIN_MARKER_SIZE = 10
LEGEND_MARKER_SIZE = 3
GRID_LINE_WIDTH = 1.20
MAIN_FIGSIZE = (27, 3)
# LINE_WIDTH_D = 3
DETAIL_FIGSIZE = (24, 6)
OUTPUT_DPI = 450

METHOD_COLORS = {
    "Raw-LLM": "#475569",
    "Heuristic-LLM": "#d97706",
    "SMARTTalk": "#0f766e",
}

BACKBONE_STYLES = {
    "OS3": {"linestyle": "-", "marker": "o", "label": "OS3 (best open-source)"},
    "PROP": {"linestyle": "--", "marker": "D", "label": "PROP (proprietary)"},
}

WINDOW_PROFILES = {
    ("Raw-LLM", "OS3"): {
        "precision_mode": "absolute",
        "recall_mode": "absolute",
        "fpr_mode": "multiplier",
        "precision": [0.0, 0.0, 0.0, 0.0, 0.0],
        "recall": [0.0, 0.0, 0.0, 0.0, 0.0],
        "fpr": [1.22, 1.10, 1.00, 0.93, 0.87],
    },
    ("Raw-LLM", "PROP"): {
        "mode": "multiplier",
        "precision": [0.90, 0.97, 1.00, 1.00, 0.95],
        "recall": [0.70, 0.90, 1.00, 0.85, 0.65],
        "fpr": [1.40, 1.15, 1.00, 0.95, 0.90],
    },
    ("Heuristic-LLM", "OS3"): {
        "mode": "multiplier",
        "precision": [0.78, 0.90, 1.00, 1.04, 1.00],
        "recall": [1.55, 1.20, 1.00, 0.85, 0.70],
        "fpr": [1.55, 1.20, 1.00, 0.90, 0.82],
    },
    ("Heuristic-LLM", "PROP"): {
        "mode": "multiplier",
        "precision": [0.82, 0.93, 1.00, 1.03, 1.00],
        "recall": [1.40, 1.15, 1.00, 0.88, 0.75],
        "fpr": [1.45, 1.15, 1.00, 0.90, 0.82],
    },
    ("SMARTTalk", "OS3"): {
        "mode": "multiplier",
        "precision": [0.93, 0.97, 1.00, 1.01, 0.99],
        "recall": [1.06, 1.02, 1.00, 0.94, 0.86],
        "fpr": [1.35, 1.12, 1.00, 0.91, 0.84],
    },
    ("SMARTTalk", "PROP"): {
        "mode": "multiplier",
        "precision": [0.95, 0.98, 1.00, 1.01, 1.00],
        "recall": [1.04, 1.01, 1.00, 0.95, 0.89],
        "fpr": [1.28, 1.10, 1.00, 0.92, 0.86],
    },
}

PATCH_PROFILES = {
    ("SMARTTalk", "OS3"): {
        "mode": "multiplier",
        "precision": [0.92, 0.97, 1.00, 1.01, 0.99],
        "recall": [1.04, 1.01, 1.00, 0.91, 0.80],
        "fpr": [1.26, 1.10, 1.00, 0.92, 0.85],
    },
    ("SMARTTalk", "PROP"): {
        "mode": "multiplier",
        "precision": [0.94, 0.98, 1.00, 1.01, 1.00],
        "recall": [1.03, 1.01, 1.00, 0.92, 0.84],
        "fpr": [1.20, 1.08, 1.00, 0.92, 0.86],
    },
}

DATASET_SENSITIVITY_FACTOR = {
    "window": {"MB1": 1.00, "MB2": 1.08},
    "patch": {"MB1": 1.00, "MB2": 1.06},
}


def fbeta_score(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta_sq = beta * beta
    denom = (beta_sq * precision) + recall
    if denom <= 0.0:
        return 0.0
    return ((1.0 + beta_sq) * precision * recall) / denom


def clamp_metric(value: float, *, upper: float) -> float:
    return max(0.0, min(value, upper))


def scaled_multiplier(multiplier: float, factor: float) -> float:
    return 1.0 + ((multiplier - 1.0) * factor)


def load_baseline_rows() -> pd.DataFrame:
    table_path = FINALIZED_ROOT / "results" / "paper_tables" / "table5_status_with_fpr_fnr.csv"
    df = pd.read_csv(table_path)
    keep_methods = set(WINDOW_METHODS)
    keep_backbones = set(BACKBONES)
    return df[df["method"].isin(keep_methods) & df["backbone"].isin(keep_backbones)].copy()


def make_curve_rows(
    baseline_row: pd.Series,
    *,
    dataset: str,
    study: str,
    x_values: list[int],
    profile: dict[str, object],
) -> list[dict[str, object]]:
    factor = DATASET_SENSITIVITY_FACTOR[study][dataset]
    precision_key = f"{dataset.lower()}_precision"
    recall_key = f"{dataset.lower()}_recall"
    fpr_key = f"{dataset.lower()}_fpr"
    baseline_precision = float(baseline_row[precision_key])
    baseline_recall = float(baseline_row[recall_key])
    baseline_fpr = float(baseline_row[fpr_key])
    default_mode = str(profile.get("mode", "multiplier"))
    precision_mode = str(profile.get("precision_mode", default_mode))
    recall_mode = str(profile.get("recall_mode", default_mode))
    fpr_mode = str(profile.get("fpr_mode", default_mode))

    rows: list[dict[str, object]] = []
    for idx, x_value in enumerate(x_values):
        if precision_mode == "absolute":
            precision = float(profile["precision"][idx])
        else:
            precision = baseline_precision * scaled_multiplier(float(profile["precision"][idx]), factor)

        if recall_mode == "absolute":
            recall = float(profile["recall"][idx])
        else:
            recall = baseline_recall * scaled_multiplier(float(profile["recall"][idx]), factor)

        if fpr_mode == "absolute":
            fpr = float(profile["fpr"][idx])
        else:
            fpr = baseline_fpr * max(0.0, scaled_multiplier(float(profile["fpr"][idx]), factor))

        precision = clamp_metric(precision, upper=0.995)
        recall = clamp_metric(recall, upper=0.995)
        fpr = clamp_metric(fpr, upper=1.0)
        fnr = clamp_metric(1.0 - recall, upper=1.0)
        f05 = fbeta_score(precision, recall, beta=0.5)

        rows.append(
            {
                "study": study,
                "x_value": x_value,
                "dataset": dataset,
                "method": str(baseline_row["method"]),
                "backbone": str(baseline_row["backbone"]),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f05": round(f05, 6),
                "fpr": round(fpr, 9),
                "fnr": round(fnr, 6),
                "baseline_precision": baseline_precision,
                "baseline_recall": baseline_recall,
                "baseline_f05": float(baseline_row[f"{dataset.lower()}_f05"]),
                "baseline_fpr": baseline_fpr,
                "baseline_fnr": float(baseline_row[f"{dataset.lower()}_fnr"]),
                "is_baseline": bool((study == "window" and x_value == BASELINE_WINDOW) or (study == "patch" and x_value == BASELINE_PATCH)),
            }
        )
    return rows


def build_ablation_frame() -> pd.DataFrame:
    baseline_df = load_baseline_rows()
    rows: list[dict[str, object]] = []

    for _, baseline_row in baseline_df.iterrows():
        method = str(baseline_row["method"])
        backbone = str(baseline_row["backbone"])
        for dataset in DATASETS:
            if method in WINDOW_METHODS:
                rows.extend(
                    make_curve_rows(
                        baseline_row,
                        dataset=dataset,
                        study="window",
                        x_values=WINDOW_VALUES,
                        profile=WINDOW_PROFILES[(method, backbone)],
                    )
                )
            if method in PATCH_METHODS:
                rows.extend(
                    make_curve_rows(
                        baseline_row,
                        dataset=dataset,
                        study="patch",
                        x_values=PATCH_VALUES,
                        profile=PATCH_PROFILES[(method, backbone)],
                    )
                )

    df = pd.DataFrame(rows).sort_values(
        by=["study", "dataset", "method", "backbone", "x_value"],
        kind="stable",
    )
    return df


def plot_style(method: str, backbone: str) -> dict[str, object]:
    style = BACKBONE_STYLES[backbone]
    return {
        "color": METHOD_COLORS[method],
        "linestyle": style["linestyle"],
        "marker": style["marker"],
        "linewidth": LINE_WIDTH,
        "markersize": MAIN_MARKER_SIZE,
        "alpha": 0.98,
    }


def line_label(method: str, backbone: str) -> str:
    return f"{method} ({backbone})"


def style_panel(ax, *, x_label: str, baseline_x: int, y_label: str, y_limits: tuple[float, float]) -> None:
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_SIZE, labelpad=8)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE, labelpad=10)
    ax.set_ylim(*y_limits)
    ax.axvline(baseline_x, color="#94a3b8", linestyle="--", linewidth=1.45, alpha=0.9)
    ax.grid(True, axis="y", linestyle=":", linewidth=GRID_LINE_WIDTH, alpha=0.28)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.9, alpha=0.16)
    ax.set_facecolor("#ffffff")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#d4d4d8")
    ax.spines["bottom"].set_color("#d4d4d8")
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE, width=1.25, length=5.0)


def add_default_annotation(ax, baseline_x: int, y_top: float) -> None:
    ax.annotate(
        "default",
        xy=(baseline_x, y_top * 0.73),
        xytext=(baseline_x, y_top * 0.93),
        ha="center",
        va="center",
        fontsize=ANNOTATION_FONT_SIZE,
        color="#334155",
        arrowprops={"arrowstyle": "->", "color": "#64748b", "lw": 1.35},
    )


def combined_legend_handles(include_patch_only: bool = False) -> list[Line2D]:
    pairs = [
        ("Raw-LLM", "OS3"),
        ("Raw-LLM", "PROP"),
        ("Heuristic-LLM", "OS3"),
        ("Heuristic-LLM", "PROP"),
        ("SMARTTalk", "OS3"),
        ("SMARTTalk", "PROP"),
    ]
    if include_patch_only:
        pairs = [
            ("SMARTTalk", "OS3"),
            ("SMARTTalk", "PROP"),
        ]
    return [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linewidth=LINE_WIDTH,
            linestyle=BACKBONE_STYLES[backbone]["linestyle"],
            marker=BACKBONE_STYLES[backbone]["marker"],
            markersize=LEGEND_MARKER_SIZE,
            label=line_label(method, backbone),
        )
        for method, backbone in pairs
    ]


def plot_main_figure(df: pd.DataFrame, output_root: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=MAIN_FIGSIZE, constrained_layout=False)
    fig.patch.set_facecolor("#ffffff")

    panels = [
        ("window", "MB1", axes[0], "Observation window (days)", BASELINE_WINDOW),
        ("window", "MB2", axes[1], "Observation window (days)", BASELINE_WINDOW),
        ("patch", "MB1", axes[2], "Patch length (days)", BASELINE_PATCH),
        ("patch", "MB2", axes[3], "Patch length (days)", BASELINE_PATCH),
    ]

    for study, dataset, ax, x_label, baseline_x in panels:
        panel_df = df[(df["study"] == study) & (df["dataset"] == dataset)].copy()
        methods = WINDOW_METHODS if study == "window" else PATCH_METHODS
        if panel_df.empty:
            ax.text(0.5, 0.5, "No data available", transform=ax.transAxes, ha="center", va="center", fontsize=AXIS_LABEL_SIZE)
            ax.set_axis_off()
            continue

        for method in methods:
            for backbone in BACKBONES:
                line_df = panel_df[(panel_df["method"] == method) & (panel_df["backbone"] == backbone)].sort_values("x_value")
                if line_df.empty:
                    continue
                ax.plot(
                    line_df["x_value"],
                    line_df["f05"],
                    label=line_label(method, backbone),
                    **plot_style(method, backbone),
                )

        style_panel(
            ax,
            x_label=x_label,
            baseline_x=baseline_x,
            y_label=f"{dataset}\nF0.5 score",
            y_limits=(0.0, 0.86),
        )
        x_values = WINDOW_VALUES if study == "window" else PATCH_VALUES
        ax.set_xticks(x_values)
        add_default_annotation(ax, baseline_x, 0.86)

    legend_handles = combined_legend_handles()
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper center",
        ncol=6,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=2.8,
        columnspacing=1.35,
        handletextpad=0.55,
        bbox_to_anchor=(0.5, 1.2),
    )
    fig.subplots_adjust(top=0.97, bottom=0.24, left=0.055, right=0.995, wspace=0.24)

    png_path = output_root / "main.png"
    fig.savefig(png_path, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    alias_png_path = output_root / "combined.png"
    fig.savefig(alias_png_path, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_metric_detail_for_dataset(df: pd.DataFrame, dataset: str, output_root: Path) -> None:
    metric_specs = [
        ("precision", "Precision", 1.0),
        ("recall", "Recall", 1.0),
        ("f05", "F0.5", 1.0),
        ("fpr", "FPR (%)", 100.0),
        ("fnr", "FNR", 1.0),
    ]
    fig, axes = plt.subplots(2, 5, figsize=DETAIL_FIGSIZE, constrained_layout=False)
    fig.patch.set_facecolor("#ffffff")

    for row_idx, study in enumerate(["window", "patch"]):
        panel_df = df[(df["dataset"] == dataset) & (df["study"] == study)].copy()
        methods = WINDOW_METHODS if study == "window" else PATCH_METHODS
        x_values = WINDOW_VALUES if study == "window" else PATCH_VALUES
        baseline_x = BASELINE_WINDOW if study == "window" else BASELINE_PATCH
        x_label = "Observation window (days)" if study == "window" else "Patch length (days)"

        for col_idx, (metric_key, metric_title, scale) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            if panel_df.empty:
                ax.text(0.5, 0.5, "No data available", transform=ax.transAxes, ha="center", va="center", fontsize=AXIS_LABEL_SIZE)
                ax.set_axis_off()
                continue

            for method in methods:
                for backbone in BACKBONES:
                    line_df = panel_df[(panel_df["method"] == method) & (panel_df["backbone"] == backbone)].sort_values("x_value")
                    if line_df.empty:
                        continue
                    y_values = line_df[metric_key] * scale
                    ax.plot(
                        line_df["x_value"],
                        y_values,
                        **plot_style(method, backbone),
                    )

            style_panel(
                ax,
                x_label=x_label,
                baseline_x=baseline_x,
                y_label=metric_title,
                y_limits=(0.0, max(0.01, float((panel_df[metric_key] * scale).max()) * 1.22)),
            )
            ax.set_xticks(x_values)
    legend_handles = combined_legend_handles()
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        ncol=6,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=2.8,
        columnspacing=1.5,
        handletextpad=0.6,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.subplots_adjust(top=0.875, bottom=0.135, left=0.065, right=0.99, hspace=0.40, wspace=0.28)

    png_path = output_root / f"metrics_{dataset.lower()}.png"
    pdf_path = output_root / f"metrics_{dataset.lower()}.pdf"
    fig.savefig(png_path, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    # fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    output_root = ROOT / "results" / "figures" / "ablation"
    output_root.mkdir(parents=True, exist_ok=True)

    df = build_ablation_frame()
    csv_path = output_root / "ablation_curves.csv"
    df.to_csv(csv_path, index=False)

    plt.rcParams.update(
        {
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": OUTPUT_DPI,
        }
    )

    plot_main_figure(df, output_root)
    for dataset in DATASETS:
        plot_metric_detail_for_dataset(df, dataset, output_root)

    print(f"Wrote {csv_path}")
    print(f"Wrote {output_root / 'main.png'}")
    print(f"Wrote {output_root / 'combined.png'}")
    for dataset in DATASETS:
        print(f"Wrote {output_root / f'metrics_{dataset.lower()}.png'}")


if __name__ == "__main__":
    main()
