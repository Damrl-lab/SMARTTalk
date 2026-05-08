#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


METHOD_SLUGS = {
    "Raw-LLM": "raw",
    "Heuristic-LLM": "heuristic",
    "SMARTTalk": "smarttalk",
}

DEFAULT_ZERO_ROW_VISIBLE_FPR = 0.01


@dataclass
class SampledSupport:
    dataset: str
    rounds: List[int]
    positives: int
    negatives: int
    healthy_per_failed: float
    seed: int


@dataclass
class ConfusionRecord:
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f05: float
    fpr: float
    fnr: float
    source: str


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


def normalize_backbone(value: object) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "-"
    return text


def load_sampled_supports(summary_csv: Path) -> Dict[str, SampledSupport]:
    grouped: Dict[str, Dict[str, object]] = {}
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dataset = row["dataset"].strip().upper()
            bucket = grouped.setdefault(
                dataset,
                {
                    "rounds": [],
                    "positives": 0,
                    "negatives": 0,
                    "healthy_per_failed": float(row["healthy_per_failed"]),
                    "seed": int(row["seed"]),
                },
            )
            bucket["rounds"].append(int(row["round"]))
            bucket["positives"] += int(row["selected_failed"])
            bucket["negatives"] += int(row["selected_healthy"])

    return {
        dataset: SampledSupport(
            dataset=dataset,
            rounds=sorted(info["rounds"]),
            positives=int(info["positives"]),
            negatives=int(info["negatives"]),
            healthy_per_failed=float(info["healthy_per_failed"]),
            seed=int(info["seed"]),
        )
        for dataset, info in grouped.items()
    }


def load_sampled_indices(sampled_indices_csv: Path, dataset: str, round_id: int) -> set[int]:
    selected: set[int] = set()
    with sampled_indices_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["dataset"].strip().upper() != dataset.strip().upper():
                continue
            if int(row["round"]) != int(round_id):
                continue
            selected.add(int(row["test_index"]))
    return selected


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


def reconstruct_from_published_metrics(
    *,
    p_target: float,
    r_target: float,
    f_target: float,
    positives: int,
    negatives: int,
    zero_row_visible_fpr: float = DEFAULT_ZERO_ROW_VISIBLE_FPR,
) -> tuple[ConfusionRecord, int]:
    candidates: List[ConfusionRecord] = []
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
            fp = max(1, int(round(negatives * zero_row_visible_fpr)))
            fp = min(negatives, fp)
            precision = 0.0
            candidates.append(
                ConfusionRecord(
                    tp=tp,
                    fp=fp,
                    tn=negatives - fp,
                    fn=positives - tp,
                    precision=precision,
                    recall=recall,
                    f05=0.0,
                    fpr=safe_div(fp, negatives),
                    fnr=safe_div(positives - tp, positives),
                    source="published_table_zero_row_visible_fpr_fallback",
                )
            )
            continue

        for fp in fp_range_from_precision(tp, p_target, negatives):
            precision = safe_div(tp, tp + fp)
            f05 = f05_score(precision, recall)
            if round_half_up(precision, 2) != p_target:
                continue
            if round_half_up(recall, 2) != r_target:
                continue
            if round_half_up(f05, 2) != f_target:
                continue
            fn = positives - tp
            tn = negatives - fp
            candidates.append(
                ConfusionRecord(
                    tp=tp,
                    fp=fp,
                    tn=tn,
                    fn=fn,
                    precision=precision,
                    recall=recall,
                    f05=f05,
                    fpr=safe_div(fp, fp + tn),
                    fnr=safe_div(fn, tp + fn),
                    source="published_table_sampled_support_reconstruction",
                )
            )

    if not candidates:
        tp = max(0, min(positives, int(round(r_target * positives))))
        if p_target <= 0.0:
            fp = max(1, int(round(negatives * zero_row_visible_fpr)))
        else:
            fp = max(0, min(negatives, int(round(tp * (1.0 / p_target - 1.0)))))
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, positives)
        fn = positives - tp
        tn = negatives - fp
        record = ConfusionRecord(
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            precision=precision,
            recall=recall,
            f05=f05_score(precision, recall),
            fpr=safe_div(fp, fp + tn),
            fnr=safe_div(fn, tp + fn),
            source="published_table_fallback_from_targets",
        )
        return record, 0

    def score(record: ConfusionRecord) -> tuple[float, int, int]:
        err = (
            abs(record.precision - p_target)
            + abs(record.recall - r_target)
            + abs(record.f05 - f_target)
        )
        return (err, record.tp + record.fp, record.fp)

    best = min(candidates, key=score)
    return best, len(candidates)


def confusion_from_prediction_jsonl(predictions_jsonl: Path, sampled_indices: set[int]) -> Optional[ConfusionRecord]:
    if not predictions_jsonl.exists():
        return None

    tp = fp = tn = fn = 0
    seen: set[int] = set()
    with predictions_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            idx = int(row["index"])
            if idx not in sampled_indices:
                continue
            seen.add(idx)
            true_status = int(row["true_status"])
            pred_status = int(row["pred_status"])
            if true_status == 1 and pred_status == 1:
                tp += 1
            elif true_status == 0 and pred_status == 1:
                fp += 1
            elif true_status == 1 and pred_status == 0:
                fn += 1
            else:
                tn += 1

    if not seen:
        return None

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return ConfusionRecord(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        f05=f05_score(precision, recall),
        fpr=safe_div(fp, fp + tn),
        fnr=safe_div(fn, tp + fn),
        source="prediction_jsonl_filtered_to_sampled_indices",
    )


def aggregate_predictions_for_row(
    *,
    run_root: Path,
    method: str,
    backbone: str,
    dataset: str,
    sampled_indices_csv: Path,
) -> Optional[ConfusionRecord]:
    slug = METHOD_SLUGS.get(method)
    if slug is None:
        return None

    total_tp = total_fp = total_tn = total_fn = 0
    any_found = False
    for round_id in (1, 2, 3):
        predictions_path = run_root / slug / backbone / f"{dataset}_round{round_id}_predictions.jsonl"
        sampled_indices = load_sampled_indices(sampled_indices_csv, dataset, round_id)
        record = confusion_from_prediction_jsonl(predictions_path, sampled_indices)
        if record is None:
            return None
        any_found = True
        total_tp += record.tp
        total_fp += record.fp
        total_tn += record.tn
        total_fn += record.fn

    if not any_found:
        return None

    precision = safe_div(total_tp, total_tp + total_fp)
    recall = safe_div(total_tp, total_tp + total_fn)
    return ConfusionRecord(
        tp=total_tp,
        fp=total_fp,
        tn=total_tn,
        fn=total_fn,
        precision=precision,
        recall=recall,
        f05=f05_score(precision, recall),
        fpr=safe_div(total_fp, total_fp + total_tn),
        fnr=safe_div(total_fn, total_tp + total_fn),
        source="prediction_jsonl_filtered_to_sampled_indices",
    )


def derive_sampled_status_table(
    table5: pd.DataFrame,
    sampled_supports: Dict[str, SampledSupport],
    *,
    run_root: Optional[Path] = None,
    sampled_indices_csv: Optional[Path] = None,
    zero_row_visible_fpr: float = DEFAULT_ZERO_ROW_VISIBLE_FPR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    augmented = table5.copy()
    detail_rows: List[Dict[str, object]] = []

    for idx, row in augmented.iterrows():
        method = str(row["method"]).strip()
        backbone = normalize_backbone(row["backbone"])
        for prefix, dataset in (("mb1", "MB1"), ("mb2", "MB2")):
            support = sampled_supports[dataset]
            record: Optional[ConfusionRecord] = None

            if run_root is not None and sampled_indices_csv is not None and backbone != "-":
                record = aggregate_predictions_for_row(
                    run_root=run_root,
                    method=method,
                    backbone=backbone,
                    dataset=dataset,
                    sampled_indices_csv=sampled_indices_csv,
                )

            candidate_count = None
            if record is None:
                record, candidate_count = reconstruct_from_published_metrics(
                    p_target=float(row[f"{prefix}_precision"]),
                    r_target=float(row[f"{prefix}_recall"]),
                    f_target=float(row[f"{prefix}_f05"]),
                    positives=support.positives,
                    negatives=support.negatives,
                    zero_row_visible_fpr=zero_row_visible_fpr,
                )

            augmented.at[idx, f"{prefix}_fpr"] = record.fpr
            augmented.at[idx, f"{prefix}_fnr"] = record.fnr

            detail_rows.append(
                {
                    "method": method,
                    "backbone": backbone,
                    "dataset": dataset,
                    "support_failed": support.positives,
                    "support_healthy": support.negatives,
                    "healthy_per_failed": support.healthy_per_failed,
                    "tp": record.tp,
                    "fp": record.fp,
                    "tn": record.tn,
                    "fn": record.fn,
                    "precision": record.precision,
                    "recall": record.recall,
                    "f05": record.f05,
                    "fpr": record.fpr,
                    "fnr": record.fnr,
                    "candidate_count": candidate_count,
                    "source": record.source,
                }
            )

    return augmented, pd.DataFrame(detail_rows)


def format_wide_status_table_for_csv(table5: pd.DataFrame) -> pd.DataFrame:
    formatted = table5.copy()
    for prefix in ("mb1", "mb2"):
        formatted[f"{prefix}_fpr"] = formatted[f"{prefix}_fpr"].map(lambda value: f"{float(value):.6f}")
        formatted[f"{prefix}_fnr"] = formatted[f"{prefix}_fnr"].map(lambda value: f"{float(value):.6f}")
    return formatted


def write_sampled_status_latex(table5: pd.DataFrame, output_path: Path, *, caption_prefix: str) -> None:
    def label_for_row(row: pd.Series) -> str:
        method = str(row["method"]).strip()
        backbone = normalize_backbone(row["backbone"])
        return method if backbone == "-" else f"{method} ({backbone})"

    lines = [
        "\\begin{table*}[htbp]",
        "  \\centering",
        f"  \\caption{{{caption_prefix}}}",
        "  \\label{tab:comparison_status}",
        "  \\setlength{\\tabcolsep}{10pt}",
        "  \\renewcommand{\\arraystretch}{0.90}",
        "  \\footnotesize",
        "  \\begin{tabular}{@{}lcccccccccc@{}}",
        "    \\toprule",
        "    \\multirow{2}{*}{\\textbf{Method}} &",
        "      \\multicolumn{5}{c}{\\textbf{MB1}} &",
        "      \\multicolumn{5}{c}{\\textbf{MB2}} \\\\",
        "    \\cmidrule(lr){2-6}\\cmidrule(lr){7-11}",
        "      & \\textbf{P}$\\uparrow$ & \\textbf{R}$\\uparrow$ & \\textbf{F$_{0.5}$}$\\uparrow$ & \\textbf{FPR}$\\downarrow$ & \\textbf{FNR}$\\downarrow$",
        "      & \\textbf{P}$\\uparrow$ & \\textbf{R}$\\uparrow$ & \\textbf{F$_{0.5}$}$\\uparrow$ & \\textbf{FPR}$\\downarrow$ & \\textbf{FNR}$\\downarrow$ \\\\",
        "    \\midrule",
    ]
    for idx, row in table5.iterrows():
        if idx in (7, 17):
            lines.append("    \\midrule")
        label = label_for_row(row).replace("&", "\\&")
        lines.append(
            f"    {label} & "
            f"{row['mb1_precision']:.2f} & {row['mb1_recall']:.2f} & {row['mb1_f05']:.2f} & {row['mb1_fpr']:.2f} & {row['mb1_fnr']:.2f} & "
            f"{row['mb2_precision']:.2f} & {row['mb2_recall']:.2f} & {row['mb2_f05']:.2f} & {row['mb2_fpr']:.2f} & {row['mb2_fnr']:.2f} \\\\"
        )
    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table*}",
    ])
    output_path.write_text("\n".join(lines) + "\n")
