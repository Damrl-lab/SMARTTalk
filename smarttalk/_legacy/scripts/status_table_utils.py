#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


TABLE5_DATASETS = {
    "mb1": ("MB1", "mb1_precision", "mb1_recall", "mb1_f05"),
    "mb2": ("MB2", "mb2_precision", "mb2_recall", "mb2_f05"),
}


def round_half_up(value: float, digits: int = 2) -> float:
    q = Decimal("1").scaleb(-digits)
    return float(Decimal(str(value)).quantize(q, rounding=ROUND_HALF_UP))


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def f05_score(precision: float, recall: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = 0.25
    return (1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def assumed_ratio_fpr(precision: float, recall: float, healthy_per_fail: float) -> float:
    if healthy_per_fail <= 0.0:
        raise ValueError("healthy_per_fail must be positive.")
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    return max(0.0, recall * (1.0 / precision - 1.0) / healthy_per_fail)


@dataclass
class DatasetSupport:
    dataset: str
    rounds: List[int]
    positives: int
    negatives: int

    @property
    def negative_to_positive_ratio(self) -> float:
        return safe_div(self.negatives, self.positives)


@dataclass
class ConfusionCandidate:
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f05: float

    @property
    def fpr(self) -> float:
        return safe_div(self.fp, self.fp + self.tn)

    @property
    def fnr(self) -> float:
        return safe_div(self.fn, self.tp + self.fn)


def load_dataset_supports(processed_root: Path) -> Dict[str, DatasetSupport]:
    supports: Dict[str, DatasetSupport] = {}
    for dataset in ("MB1", "MB2"):
        rounds: List[int] = []
        positives = 0
        negatives = 0
        for round_id in (1, 2, 3):
            path = processed_root / f"{dataset}_round{round_id}" / "test.npz"
            data = np.load(path)
            y = np.asarray(data["y"]).astype(int)
            rounds.append(round_id)
            positives += int((y == 1).sum())
            negatives += int((y == 0).sum())
        supports[dataset] = DatasetSupport(
            dataset=dataset,
            rounds=rounds,
            positives=positives,
            negatives=negatives,
        )
    return supports


def fp_range_from_precision(tp: int, p_target: float, negatives: int) -> Iterable[int]:
    if p_target == 0.0:
        if tp == 0:
            yield 0
        return

    lo = max(0.0, p_target - 0.005)
    hi = min(1.0, p_target + 0.005)

    if hi >= 1.0:
        min_fp = 0
    else:
        min_fp = max(0, math.ceil(tp / hi - tp))

    if lo <= 0.0:
        max_fp = negatives
    else:
        max_fp = min(negatives, max(0, math.floor(tp / lo - tp - 1e-12)))

    for fp in range(min_fp, max_fp + 1):
        yield fp


def reconstruct_confusion_matrix(
    p_target: float,
    r_target: float,
    f_target: float,
    positives: int,
    negatives: int,
) -> tuple[ConfusionCandidate, int, str, float, float]:
    candidates: List[ConfusionCandidate] = []

    tp_candidates = [
        tp for tp in range(positives + 1)
        if round_half_up(safe_div(tp, positives), 2) == r_target
    ]

    if not tp_candidates:
        tp_guess = int(round(r_target * positives))
        tp_candidates = [max(0, min(positives, tp_guess))]

    for tp in tp_candidates:
        recall = safe_div(tp, positives)

        if p_target == 0.0 and r_target == 0.0 and f_target == 0.0:
            fp_values = [0]
        else:
            fp_values = fp_range_from_precision(tp, p_target, negatives)

        for fp in fp_values:
            precision = safe_div(tp, tp + fp)
            f05 = f05_score(precision, recall)

            if round_half_up(precision, 2) != p_target:
                continue
            if round_half_up(recall, 2) != r_target:
                continue
            if round_half_up(f05, 2) != f_target:
                continue

            candidates.append(
                ConfusionCandidate(
                    tp=tp,
                    fp=fp,
                    fn=positives - tp,
                    tn=negatives - fp,
                    precision=precision,
                    recall=recall,
                    f05=f05,
                )
            )

    if not candidates:
        tp = max(0, min(positives, int(round(r_target * positives))))
        if p_target <= 0.0:
            fp = 0
        else:
            fp = max(0, min(negatives, int(round(tp * (1.0 / p_target - 1.0)))))
        cand = ConfusionCandidate(
            tp=tp,
            fp=fp,
            fn=positives - tp,
            tn=negatives - fp,
            precision=safe_div(tp, tp + fp),
            recall=safe_div(tp, positives),
            f05=f05_score(safe_div(tp, tp + fp), safe_div(tp, positives)),
        )
        return cand, 0, "fallback_from_targets", cand.fpr, cand.fpr

    def score(cand: ConfusionCandidate) -> tuple[float, int, int]:
        err = (
            abs(cand.precision - p_target)
            + abs(cand.recall - r_target)
            + abs(cand.f05 - f_target)
        )
        return (err, cand.tp + cand.fp, cand.fp)

    best = min(candidates, key=score)
    fpr_values = [cand.fpr for cand in candidates]
    return best, len(candidates), "rounded_table_reconstruction", min(fpr_values), max(fpr_values)


def derive_table5_with_rates(
    table5: pd.DataFrame,
    processed_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, DatasetSupport]]:
    supports = load_dataset_supports(processed_root)
    augmented = table5.copy()
    detail_rows: List[Dict[str, object]] = []

    for idx, row in augmented.iterrows():
        for prefix, (dataset, p_col, r_col, f_col) in TABLE5_DATASETS.items():
            support = supports[dataset]
            confusion, candidate_count, source, fpr_lower, fpr_upper = reconstruct_confusion_matrix(
                p_target=float(row[p_col]),
                r_target=float(row[r_col]),
                f_target=float(row[f_col]),
                positives=support.positives,
                negatives=support.negatives,
            )

            augmented.at[idx, f"{prefix}_fpr"] = round(confusion.fpr, 6)
            augmented.at[idx, f"{prefix}_fnr"] = round(1.0 - float(row[r_col]), 2)

            detail_rows.append(
                {
                    "method": row["method"],
                    "backbone": row["backbone"],
                    "dataset": dataset,
                    "support_pos": support.positives,
                    "support_neg": support.negatives,
                    "negative_to_positive_ratio": support.negative_to_positive_ratio,
                    "tp": confusion.tp,
                    "fp": confusion.fp,
                    "tn": confusion.tn,
                    "fn": confusion.fn,
                    "precision": confusion.precision,
                    "recall": confusion.recall,
                    "f05": confusion.f05,
                    "fpr": confusion.fpr,
                    "fpr_lower": fpr_lower,
                    "fpr_upper": fpr_upper,
                    "fpr_span": fpr_upper - fpr_lower,
                    "fnr": confusion.fnr,
                    "fnr_from_reported_recall": 1.0 - float(row[r_col]),
                    "candidate_count": candidate_count,
                    "reconstruction_source": source,
                }
            )

    return augmented, pd.DataFrame(detail_rows), supports


def derive_table5_with_assumed_ratio(
    table5: pd.DataFrame,
    healthy_per_fail: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    augmented = table5.copy()
    detail_rows: List[Dict[str, object]] = []

    for idx, row in augmented.iterrows():
        for prefix, (dataset, p_col, r_col, f_col) in TABLE5_DATASETS.items():
            precision = float(row[p_col])
            recall = float(row[r_col])
            f05 = float(row[f_col])
            fpr = assumed_ratio_fpr(precision, recall, healthy_per_fail)
            fnr = 1.0 - recall

            augmented.at[idx, f"{prefix}_fpr"] = round(fpr, 6)
            augmented.at[idx, f"{prefix}_fnr"] = round(fnr, 2)

            detail_rows.append(
                {
                    "method": row["method"],
                    "backbone": row["backbone"],
                    "dataset": dataset,
                    "assumed_healthy_per_fail": healthy_per_fail,
                    "assumed_failed_fraction": 1.0 / (1.0 + healthy_per_fail),
                    "assumed_healthy_fraction": healthy_per_fail / (1.0 + healthy_per_fail),
                    "precision": precision,
                    "recall": recall,
                    "f05": f05,
                    "fpr": fpr,
                    "fnr": fnr,
                    "derivation": "fpr = recall * (1/precision - 1) / healthy_per_fail",
                }
            )

    return augmented, pd.DataFrame(detail_rows)


def summarize_rate_reconstruction(detail_rows: pd.DataFrame) -> Dict[str, object]:
    if detail_rows.empty:
        return {
            "row_count": 0,
            "ambiguous_row_count": 0,
            "max_fpr_span": 0.0,
            "worst_case_row": None,
        }

    detail = detail_rows.copy()
    detail["candidate_count"] = detail["candidate_count"].astype(int)
    detail["fpr_span"] = detail["fpr_span"].astype(float)

    ambiguous = detail[detail["candidate_count"] > 1]
    if ambiguous.empty:
        worst_case = None
        max_fpr_span = 0.0
    else:
        worst = ambiguous.sort_values("fpr_span", ascending=False).iloc[0]
        max_fpr_span = float(worst["fpr_span"])
        worst_case = {
            "method": str(worst["method"]),
            "backbone": normalize_backbone(worst["backbone"]),
            "dataset": str(worst["dataset"]),
            "candidate_count": int(worst["candidate_count"]),
            "selected_fpr": float(worst["fpr"]),
            "fpr_lower": float(worst["fpr_lower"]),
            "fpr_upper": float(worst["fpr_upper"]),
            "fpr_span": max_fpr_span,
        }

    return {
        "row_count": int(len(detail)),
        "ambiguous_row_count": int(len(ambiguous)),
        "max_fpr_span": max_fpr_span,
        "worst_case_row": worst_case,
    }


def ratio_assumption_payload(healthy_per_fail: float) -> Dict[str, object]:
    return {
        "healthy_per_fail": healthy_per_fail,
        "healthy_to_failed_ratio_text": f"{healthy_per_fail:.2f}:1",
        "failed_to_healthy_ratio_text": f"1:{healthy_per_fail:.2f}",
        "assumed_failed_fraction": 1.0 / (1.0 + healthy_per_fail),
        "assumed_healthy_fraction": healthy_per_fail / (1.0 + healthy_per_fail),
        "fpr_formula": "FPR = Recall * (1 / Precision - 1) / healthy_per_fail",
        "fnr_formula": "FNR = 1 - Recall",
    }


def supports_payload(supports: Dict[str, DatasetSupport]) -> Dict[str, Dict[str, object]]:
    payload: Dict[str, Dict[str, object]] = {}
    for dataset, support in supports.items():
        payload[dataset] = {
            "rounds": support.rounds,
            "positives": support.positives,
            "negatives": support.negatives,
            "negative_to_positive_ratio": support.negative_to_positive_ratio,
        }
    return payload


def normalize_backbone(value: object) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "-"
    return text


def write_table5_with_rates_latex(
    table5: pd.DataFrame,
    output_path: Path,
    healthy_per_fail: float = 1.0,
) -> None:
    def method_label(row: pd.Series) -> str:
        method = str(row["method"]).strip()
        backbone = normalize_backbone(row["backbone"])
        label = method if backbone == "-" else f"{method} ({backbone})"
        return label.replace("&", "\\&")

    if abs(healthy_per_fail - 1.0) < 1e-9:
        ratio_note = "1:1 healthy:failed sampled evaluation ratio"
        fpr_formula = "\\mathrm{FPR}=R\\,(1/P-1)"
    else:
        ratio_note = f"{healthy_per_fail:g}:1 healthy:failed sampled evaluation ratio"
        fpr_formula = f"\\mathrm{{FPR}}=R\\,(1/P-1)/{healthy_per_fail:g}"

    lines = [
        f"% Estimated under an assumed {ratio_note}.",
        "\\begin{table*}[htbp]",
        "  \\centering",
        f"  \\caption{{Status prediction performance. Precision (P), Recall (R), F$_{{0.5}}$, false positive rate (FPR), and false negative rate (FNR) are reported. FNR is computed as $1-R$. FPR is estimated under an assumed {ratio_note} using ${fpr_formula}$.}}",
        "  \\label{tab:comparison_status}",
        "  \\setlength{\\tabcolsep}{10pt}",
        "  \\renewcommand{\\arraystretch}{0.90}",
        "  \\footnotesize",
        "  \\begin{tabular}{@{}lcccccccccc@{}}",
        "\\toprule",
        "\\multirow{2}{*}{\\textbf{Method}} & \\multicolumn{5}{c}{\\textbf{MB1}} & \\multicolumn{5}{c}{\\textbf{MB2}}\\\\",
        "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}",
        " & \\textbf{P}$\\uparrow$ & \\textbf{R}$\\uparrow$ & \\textbf{F$_{0.5}$}$\\uparrow$ & \\textbf{FPR}$\\downarrow$ & \\textbf{FNR}$\\downarrow$ & \\textbf{P}$\\uparrow$ & \\textbf{R}$\\uparrow$ & \\textbf{F$_{0.5}$}$\\uparrow$ & \\textbf{FPR}$\\downarrow$ & \\textbf{FNR}$\\downarrow$\\\\",
        "\\midrule",
    ]
    for idx, row in table5.iterrows():
        label = method_label(row)
        if idx in (7, 17):
            lines.append("\\midrule")
        lines.append(
            f"{label} & "
            f"{row['mb1_precision']:.2f} & {row['mb1_recall']:.2f} & {row['mb1_f05']:.2f} & {row['mb1_fpr']:.2f} & {row['mb1_fnr']:.2f} & "
            f"{row['mb2_precision']:.2f} & {row['mb2_recall']:.2f} & {row['mb2_f05']:.2f} & {row['mb2_fpr']:.2f} & {row['mb2_fnr']:.2f}\\\\"
        )
    lines.extend([
        "\\bottomrule",
        "  \\end{tabular}",
        "\\end{table*}",
    ])
    output_path.write_text("\n".join(lines) + "\n")


def format_table5_with_rates_for_csv(table5: pd.DataFrame) -> pd.DataFrame:
    formatted = table5.copy()
    for prefix in ("mb1", "mb2"):
        formatted[f"{prefix}_fpr"] = formatted[f"{prefix}_fpr"].map(lambda value: f"{value:.6f}")
        formatted[f"{prefix}_fnr"] = formatted[f"{prefix}_fnr"].map(lambda value: f"{value:.2f}")
    return formatted
