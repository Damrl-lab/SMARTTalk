#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from sensitivity_common import DEFAULT_PATCH_VALUES, DEFAULT_WINDOW_VALUES, ROOT


WINDOW_METHODS = ["Raw-LLM", "Heuristic-LLM", "SMARTTalk"]
PATCH_METHODS = ["SMARTTalk"]
BACKBONES = ["OS3", "PROP"]
DATASETS = ["MB1", "MB2"]
BASELINE_WINDOW = 30
BASELINE_PATCH = 5

METHOD_COLORS = {
    "Raw-LLM": "#475569",
    "Heuristic-LLM": "#d97706",
    "SMARTTalk": "#0f766e",
}

BACKBONE_STYLES = {
    "OS3": {"linestyle": "-", "marker": "o", "label": "OS3 (best open-source)"},
    "PROP": {"linestyle": "--", "marker": "D", "label": "PROP (proprietary)"},
}


def plot_style(method: str, backbone: str) -> dict[str, object]:
    style = BACKBONE_STYLES[backbone]
    return {
        "color": METHOD_COLORS[method],
        "linestyle": style["linestyle"],
        "marker": style["marker"],
        "linewidth": 2.4,
        "markersize": 6.4,
        "alpha": 0.98,
    }


def line_label(method: str, backbone: str) -> str:
    return f"{method} ({backbone})"


def load_study_frame(study: str) -> pd.DataFrame:
    study_folder = "window_sensitivity" if study == "window" else "patch_sensitivity"
    csv_path = ROOT / "results" / study_folder / "status_sensitivity_long.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    keep_methods = WINDOW_METHODS if study == "window" else PATCH_METHODS
    return df[df["method"].isin(keep_methods) & df["backbone"].isin(BACKBONES)].copy()


def style_panel(ax, *, x_label: str, baseline_x: int, y_label: str, y_limits: tuple[float, float]) -> None:
    ax.set_xlabel(x_label, fontsize=10.5)
    ax.set_ylabel(y_label, fontsize=10.5)
    ax.set_ylim(*y_limits)
    ax.axvline(baseline_x, color="#94a3b8", linestyle="--", linewidth=1.1, alpha=0.9)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.9, alpha=0.28)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.7, alpha=0.16)
    ax.set_facecolor("#ffffff")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#d4d4d8")
    ax.spines["bottom"].set_color("#d4d4d8")


def add_default_annotation(ax, baseline_x: int, y_top: float) -> None:
    ax.annotate(
        "paper default",
        xy=(baseline_x, y_top * 0.73),
        xytext=(baseline_x, y_top * 0.93),
        ha="center",
        va="center",
        fontsize=9.2,
        color="#334155",
        arrowprops={"arrowstyle": "->", "color": "#64748b", "lw": 1.0},
    )


def combined_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linewidth=2.8,
            linestyle=BACKBONE_STYLES[backbone]["linestyle"],
            marker=BACKBONE_STYLES[backbone]["marker"],
            markersize=6.2,
            label=line_label(method, backbone),
        )
        for method, backbone in [
            ("Raw-LLM", "OS3"),
            ("Raw-LLM", "PROP"),
            ("Heuristic-LLM", "OS3"),
            ("Heuristic-LLM", "PROP"),
            ("SMARTTalk", "OS3"),
            ("SMARTTalk", "PROP"),
        ]
    ]


def plot_rebuttal_main(window_df: pd.DataFrame, patch_df: pd.DataFrame, output_root: Path) -> None:
    if window_df.empty and patch_df.empty:
        print("Skipping rebuttal main plot; no sensitivity CSVs found.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.2), constrained_layout=False)
    fig.patch.set_facecolor("#ffffff")
    panels = [
        ("window", "MB1", axes[0, 0], "Observation window N (days)", BASELINE_WINDOW),
        ("window", "MB2", axes[0, 1], "Observation window N (days)", BASELINE_WINDOW),
        ("patch", "MB1", axes[1, 0], "Patch length L (days)", BASELINE_PATCH),
        ("patch", "MB2", axes[1, 1], "Patch length L (days)", BASELINE_PATCH),
    ]

    for study, dataset, ax, x_label, baseline_x in panels:
        df = window_df if study == "window" else patch_df
        methods = WINDOW_METHODS if study == "window" else PATCH_METHODS
        panel_df = df[df["dataset"] == dataset].copy()
        if panel_df.empty:
            ax.text(0.5, 0.5, "Run this study to populate this panel.", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue

        for method in methods:
            for backbone in BACKBONES:
                line_df = panel_df[(panel_df["method"] == method) & (panel_df["backbone"] == backbone)]
                if line_df.empty:
                    continue
                x_col = "window_days" if study == "window" else "patch_len"
                line_df = line_df.sort_values(by=x_col, kind="stable")
                ax.plot(
                    line_df[x_col],
                    line_df["f05"],
                    **plot_style(method, backbone),
                )

        style_panel(
            ax,
            x_label=x_label,
            baseline_x=baseline_x,
            y_label=f"{dataset}\nF0.5 score",
            y_limits=(0.0, 0.86),
        )
        ax.set_xticks(DEFAULT_WINDOW_VALUES if study == "window" else DEFAULT_PATCH_VALUES)
        add_default_annotation(ax, baseline_x, 0.86)

    legend_handles = combined_legend_handles()
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.03),
    )
    fig.subplots_adjust(top=0.97, bottom=0.14, left=0.08, right=0.985, hspace=0.24, wspace=0.18)

    png_path = output_root / "status_sensitivity_rebuttal_main.png"
    pdf_path = output_root / "status_sensitivity_rebuttal_main.pdf"
    fig.savefig(png_path, dpi=320, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def plot_metric_detail(window_df: pd.DataFrame, patch_df: pd.DataFrame, dataset: str, output_root: Path) -> None:
    if window_df.empty and patch_df.empty:
        return

    metric_specs = [
        ("precision", "Precision", 1.0),
        ("recall", "Recall", 1.0),
        ("f05", "F0.5", 1.0),
        ("fpr", "FPR (%)", 100.0),
        ("fnr", "FNR", 1.0),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(19.0, 7.8), constrained_layout=False)
    fig.patch.set_facecolor("#ffffff")

    for row_idx, study in enumerate(["window", "patch"]):
        df = window_df if study == "window" else patch_df
        panel_df = df[df["dataset"] == dataset].copy()
        methods = WINDOW_METHODS if study == "window" else PATCH_METHODS
        x_col = "window_days" if study == "window" else "patch_len"
        x_label = "Observation window N (days)" if study == "window" else "Patch length L (days)"
        baseline_x = BASELINE_WINDOW if study == "window" else BASELINE_PATCH

        for col_idx, (metric_key, metric_title, scale) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            if panel_df.empty:
                ax.text(0.5, 0.5, "Missing study outputs", transform=ax.transAxes, ha="center", va="center")
                ax.set_axis_off()
                continue

            for method in methods:
                for backbone in BACKBONES:
                    line_df = panel_df[(panel_df["method"] == method) & (panel_df["backbone"] == backbone)]
                    if line_df.empty:
                        continue
                    line_df = line_df.sort_values(by=x_col, kind="stable")
                    ax.plot(
                        line_df[x_col],
                        line_df[metric_key] * scale,
                        **plot_style(method, backbone),
                    )

            y_top = max(0.01, float((panel_df[metric_key] * scale).max()) * 1.22)
            style_panel(
                ax,
                x_label=x_label,
                baseline_x=baseline_x,
                y_label=metric_title,
                y_limits=(0.0, y_top),
            )
            ax.set_xticks(DEFAULT_WINDOW_VALUES if study == "window" else DEFAULT_PATCH_VALUES)

    legend_handles = combined_legend_handles()
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.975),
    )
    fig.subplots_adjust(top=0.90, bottom=0.11, left=0.055, right=0.99, hspace=0.34, wspace=0.22)

    png_path = output_root / f"status_sensitivity_rebuttal_metrics_{dataset.lower()}.png"
    pdf_path = output_root / f"status_sensitivity_rebuttal_metrics_{dataset.lower()}.pdf"
    fig.savefig(png_path, dpi=320, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot rebuttal-focused observation-window and patch-length sensitivity curves.",
    )
    parser.add_argument("--study", choices=["window", "patch", "both"], default="both")
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.size": 10.1,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.6,
        }
    )

    window_df = load_study_frame("window") if args.study in {"window", "both"} else pd.DataFrame()
    patch_df = load_study_frame("patch") if args.study in {"patch", "both"} else pd.DataFrame()
    output_root = ROOT / "results" / "rebuttal_plots"
    output_root.mkdir(parents=True, exist_ok=True)

    plot_rebuttal_main(window_df, patch_df, output_root)
    for dataset in DATASETS:
        plot_metric_detail(window_df, patch_df, dataset, output_root)


if __name__ == "__main__":
    main()
