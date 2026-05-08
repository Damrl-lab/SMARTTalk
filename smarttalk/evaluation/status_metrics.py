"""Confusion-matrix metrics for status prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class StatusMetrics:
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f05: float
    fpr: float
    fnr: float


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _f05(precision: float, recall: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = 0.25
    return (1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def compute_status_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> StatusMetrics:
    tp = fp = tn = fn = 0
    for truth, pred in zip(y_true, y_pred):
        if truth == 1 and pred == 1:
            tp += 1
        elif truth == 0 and pred == 1:
            fp += 1
        elif truth == 0 and pred == 0:
            tn += 1
        else:
            fn += 1
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return StatusMetrics(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        f05=_f05(precision, recall),
        fpr=_safe_div(fp, fp + tn),
        fnr=_safe_div(fn, tp + fn),
    )
