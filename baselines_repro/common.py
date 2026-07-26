"""Shared utilities for the SMART-log failure-prediction baselines: data loading,
window feature helpers, the fixed 1:23 evaluation set, and status metrics
(precision, recall, F0.5, FPR, FNR) pooled across the three temporal rounds.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

ROUNDS = (1, 2, 3)
RATIO = 23


# --------------------------------------------------------------------- data ---
def load_split(path):
    d = np.load(path)
    X = np.asarray(d["X"], np.float32)
    y = np.asarray(d["y"]).astype(int)
    ttf = np.asarray(d["ttf"]) if "ttf" in d.files else None
    feats = [f.decode() if isinstance(f, bytes) else str(f) for f in d["features"]]
    return X, y, ttf, feats


def pooled_train(data_root, dataset, rounds=ROUNDS):
    Xs, ys = [], []
    for r in rounds:
        X, y, _, _ = load_split(Path(data_root) / f"{dataset}_round{r}" / "train.npz")
        Xs.append(X); ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


def downsample(y, neg_per_pos=None, cap_neg=None, seed=0):
    """Indices keeping all positives plus a bounded number of negatives."""
    y = np.asarray(y)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    target = len(neg) if neg_per_pos is None else int(round(len(pos) * neg_per_pos))
    if cap_neg is not None:
        target = min(target, cap_neg)
    target = min(target, len(neg))
    keep = rng.choice(neg, size=target, replace=False) if target < len(neg) else neg
    idx = np.concatenate([pos, keep]); rng.shuffle(idx)
    return idx


# ----------------------------------------------------------------- features ---
def flatten(X):
    Xf = X.reshape(X.shape[0], -1)
    return np.nan_to_num(Xf, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def per_attribute_stats(X):
    """Eight summary statistics per SMART attribute over the window: min, max,
    mean, std, first-day, last-day, last-first, mean daily change."""
    Xc = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mn, mx, me, sd = Xc.min(1), Xc.max(1), Xc.mean(1), Xc.std(1)
    first, last = Xc[:, 0, :], Xc[:, -1, :]
    return np.concatenate([mn, mx, me, sd, first, last, last - first,
                           np.mean(np.diff(Xc, axis=1), axis=1)], axis=1).astype(np.float32)


class MinMaxScaler:
    def fit(self, X):
        self.mn = np.where(np.isfinite(np.nanmin(X, 0)), np.nanmin(X, 0), 0.0)
        mx = np.where(np.isfinite(np.nanmax(X, 0)), np.nanmax(X, 0), 1.0)
        rng = mx - self.mn; rng[rng < 1e-6] = 1.0; self.rng = rng
        return self
    def transform(self, X):
        return np.nan_to_num((X - self.mn) / self.rng, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ------------------------------------------------------------------ metrics ---
@dataclass
class Status:
    tp: int; fp: int; tn: int; fn: int
    precision: float; recall: float; f05: float; fpr: float; fnr: float
    def as_dict(self): return asdict(self)


def status_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int); y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1))); fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0))); fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f05 = 1.25 * p * r / (0.25 * p + r) if (p or r) else 0.0
    return Status(tp, fp, tn, fn, p, r, f05, fp / (fp + tn) if fp + tn else 0.0,
                  fn / (tp + fn) if tp + fn else 0.0)


# ------------------------------------------------------------- evaluation set --
def evaluate_on_set(model, eval_set_path):
    """Score a saved fixed 1:23 evaluation set at its decision threshold and
    return status metrics."""
    d = np.load(eval_set_path)
    X, y, tau = d["X"], d["y"].astype(int), float(d["tau"])
    return status_metrics(y, (model.score(X) >= tau).astype(int))


def _pkg_root():
    return Path(__file__).resolve().parent


def report(name, dataset, checkpoints=None, eval_sets=None):
    """Inference: load a baseline's checkpoint and report its five metrics on the
    fixed 1:23 evaluation set."""
    import joblib
    ckpt = Path(checkpoints) if checkpoints else _pkg_root() / "checkpoints" / f"{name}_{dataset}.joblib"
    es = Path(eval_sets) if eval_sets else _pkg_root() / "eval_sets" / f"{name}_{dataset}_test.npz"
    m = evaluate_on_set(joblib.load(ckpt), es)
    print(f"{name} {dataset}: P={m.precision:.2f} R={m.recall:.2f} F0.5={m.f05:.2f} "
          f"FPR={m.fpr:.4f} FNR={m.fnr:.2f}")
    return m


def train_and_save(train_fn, name, dataset, data_root, out=None):
    """Train a baseline from the data splits and save the checkpoint."""
    import joblib
    out = Path(out) if out else _pkg_root() / "checkpoints" / f"{name}_{dataset}.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    model = train_fn(data_root, dataset)
    joblib.dump(model, out, compress=3)
    print(f"trained {name} {dataset} -> {out}")
    return model


def cli(name, train_fn):
    """Command-line entry for a single baseline: inference by default (loads the
    shipped checkpoint), or `--train` to train from the data splits."""
    import argparse
    ap = argparse.ArgumentParser(description=f"{name} SMART failure-prediction baseline")
    ap.add_argument("--dataset", default="MB1", choices=["MB1", "MB2"])
    ap.add_argument("--train", action="store_true",
                    help="train from data instead of loading the shipped checkpoint")
    ap.add_argument("--data-root", default="data/splits")
    ap.add_argument("--out", default=None, help="checkpoint output path when training")
    args = ap.parse_args()
    if args.train:
        train_and_save(train_fn, name, args.dataset, args.data_root, args.out)
    else:
        report(name, args.dataset)
