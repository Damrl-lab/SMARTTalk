"""
Chen et al. (IMW'22) style baseline on 30-day SMART windows.

Implements:
  - Feature aggregation over each 30-day window (min/max/mean/std/first/last/delta/diff-mean)
  - Standardization using train split
  - Class balancing on train (1:1 healthy:failed)
  - Two models:
      * LightGBM (if installed)
      * Random Forest (sklearn)
  - Threshold tuning on a BALANCED validation subset to MAXIMIZE PRECISION
    for the failed class (label=1), as in the paper.
  - Reporting on:
      * Validation (unbalanced, full val set)
      * Test (ratio-based subset: all failed windows plus sampled healthy windows)

Input layout (same as rf_nn.py / mvtrf.py):
  data_dir/
    train.npz  (X, y, ttf, features)
    val.npz
    test.npz

  X: float32 [N, T, F]   (N windows, T=30 days, F SMART attributes)
  y: int64   [N]         (0 = healthy, 1 = failed)
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

# Optional LightGBM
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


# ----------------------------------------------------------------------
# Data loading and feature aggregation
# ----------------------------------------------------------------------


def load_split(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    X = data["X"]          # [N, T, F]
    y = data["y"]          # [N]
    ttf = data.get("ttf")  # [N] or None
    features = data["features"]
    return X, y, ttf, features


def aggregate_30day_features(X: np.ndarray) -> np.ndarray:
    """
    Aggregate each 30-day [T, F] window into a fixed-length feature vector.

    For each attribute we compute:
      - min, max, mean, std over the 30 days
      - first value, last value, delta(last - first)
      - mean daily difference (avg of X[t+1] - X[t])

    Shape:
      X: [N, T, F]
      return: [N, F * 8]
    """
    N, T, F = X.shape

    # Basic stats (handle NaNs safely)
    mins = np.nanmin(X, axis=1)          # [N, F]
    maxs = np.nanmax(X, axis=1)          # [N, F]
    means = np.nanmean(X, axis=1)        # [N, F]
    stds = np.nanstd(X, axis=1)          # [N, F]

    first = X[:, 0, :]                   # [N, F]
    last = X[:, -1, :]                   # [N, F]
    delta = last - first                 # [N, F]

    # Mean daily diff
    diffs = np.diff(X, axis=1)           # [N, T-1, F]
    diff_mean = np.nanmean(diffs, axis=1)

    feats = np.concatenate(
        [mins, maxs, means, stds, first, last, delta, diff_mean],
        axis=1,
    ).astype(np.float32)

    return feats


def standardize_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[StandardScaler, np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X_train)

    X_train_std = scaler.transform(X_train).astype(np.float32)
    X_val_std = scaler.transform(X_val).astype(np.float32)
    X_test_std = scaler.transform(X_test).astype(np.float32)

    return scaler, X_train_std, X_val_std, X_test_std


# ----------------------------------------------------------------------
# Sampling and evaluation helpers
# ----------------------------------------------------------------------


def downsample_train(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample majority class to get ~1:1 ratio (healthy vs failed) for training.
    """
    rng = check_random_state(random_state)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos == 0:
        raise ValueError("No positive (failed) samples in training set.")
    if n_neg == 0:
        raise ValueError("No negative (healthy) samples in training set.")

    if n_neg >= n_pos:
        n = n_pos
        pos_sel = pos_idx
        neg_sel = rng.choice(neg_idx, size=n, replace=False)
    else:
        n = n_neg
        pos_sel = rng.choice(pos_idx, size=n, replace=False)
        neg_sel = neg_idx

    sel_idx = np.concatenate([pos_sel, neg_sel])
    rng.shuffle(sel_idx)

    return X[sel_idx], y[sel_idx]


def balanced_indices(
    y: np.ndarray,
    random_state: int = 0,
    healthy_per_fail: float = 1.0,
) -> np.ndarray:
    """
    Build indices for a ratio-based subset:
      - use ALL failed windows
      - randomly sample `healthy_per_fail` times as many healthy windows
    """
    rng = check_random_state(random_state)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos == 0:
        raise ValueError("No positive (failed) samples.")
    if n_neg == 0:
        raise ValueError("No negative (healthy) samples.")

    target_neg = max(1, int(round(n_pos * healthy_per_fail)))
    if n_neg >= target_neg:
        pos_sel = pos_idx
        neg_sel = rng.choice(neg_idx, size=target_neg, replace=False)
    else:
        neg_sel = neg_idx
        max_pos = max(1, int(np.floor(n_neg / healthy_per_fail)))
        pos_sel = pos_idx if max_pos >= n_pos else rng.choice(pos_idx, size=max_pos, replace=False)

    sel_idx = np.concatenate([pos_sel, neg_sel])
    rng.shuffle(sel_idx)
    return sel_idx


def find_best_threshold_for_precision(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    name: str = "",
) -> Tuple[float, float, float, float]:
    """
    Search thresholds to get a high-precision, low-recall operating point
    for the failed class (label=1), similar to Chen et al.:contentReference[oaicite:1]{index=1}

    Strategy:
      - Scan thresholds in [0.05, 0.95].
      - Among thresholds with recall > 0, pick the one with max precision.
      - If all recall==0, fall back to best F1.
    """
    best_thr = 0.5
    best_prec = 0.0
    best_rec = 0.0
    best_f1 = -1.0

    candidates = []
    for thr in np.linspace(0.05, 0.95, 19):
        y_pred = (y_proba >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            pos_label=1,
            zero_division=0,
        )
        candidates.append((thr, prec, rec, f1))

    non_zero = [c for c in candidates if c[2] > 0.0]
    if non_zero:
        thr, prec, rec, f1 = max(non_zero, key=lambda x: x[1])
        best_thr, best_prec, best_rec, best_f1 = thr, prec, rec, f1
    else:
        thr, prec, rec, f1 = max(candidates, key=lambda x: x[3])
        best_thr, best_prec, best_rec, best_f1 = thr, prec, rec, f1

    print(f"\n[{name}] Threshold search on balanced validation:")
    print(
        f"  best_thr={best_thr:.3f}, "
        f"precision={best_prec:.4f}, recall={best_rec:.4f}, f1={best_f1:.4f}"
    )

    return best_thr, best_prec, best_rec, best_f1


def evaluate_split(
    name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute and print precision / recall / F1 for the failed class (label=1)
    at the given threshold.
    """
    y_pred = (y_proba >= threshold).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    print(f"\n=== {name} ===")
    print(f"Samples: {len(y_true)}")
    print(f"Precision (fail=1): {prec:.4f}")
    print(f"Recall    (fail=1): {rec:.4f}")
    print(f"F1        (fail=1): {f1:.4f}")
    print("\nClassification report (0=healthy, 1=failed):")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    return {"precision": float(prec), "recall": float(rec), "f1": float(f1)}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Chen et al. (IMW'22) style baseline on 30-day SSD windows",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="processed/MB2_round1",
        help="Directory with train.npz / val.npz / test.npz "
             "(e.g., processed/MBx_roundx)",
    )
    parser.add_argument(
        "--rf_trees",
        type=int,
        default=400,
        help="Number of trees for RandomForest",
    )
    parser.add_argument(
        "--rf_max_depth",
        type=int,
        default=9,
        help="Max depth for RandomForest (None for unlimited)",
    )
    parser.add_argument(
        "--lgbm_trees",
        type=int,
        default=400,
        help="Number of boosting rounds for LightGBM",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.npz"
    val_path = data_dir / "val.npz"
    test_path = data_dir / "test.npz"

    print(f"Loading splits from: {data_dir.resolve()}")
    X_train, y_train, ttf_train, feat_names = load_split(train_path)
    X_val,   y_val,   ttf_val,   _          = load_split(val_path)
    X_test,  y_test,  ttf_test,  _          = load_split(test_path)

    print("\nData shapes (windows × days × features):")
    print(f"  train X: {X_train.shape}, y: {y_train.shape}")
    print(f"  val   X: {X_val.shape},   y: {y_val.shape}")
    print(f"  test  X: {X_test.shape},  y: {y_test.shape}")
    print(f"Features (F={len(feat_names)}): {list(feat_names)}")

    # Aggregate each 30-day window into hand-crafted features
    print("\nAggregating 30-day windows into features...")
    X_train_agg = aggregate_30day_features(X_train)
    X_val_agg   = aggregate_30day_features(X_val)
    X_test_agg  = aggregate_30day_features(X_test)

    print(f"Aggregated feature shape:")
    print(f"  train: {X_train_agg.shape}")
    print(f"  val  : {X_val_agg.shape}")
    print(f"  test : {X_test_agg.shape}")

    # Standardize
    print("\nStandardizing features (fit on train, apply to val/test)...")
    scaler, X_train_std, X_val_std, X_test_std = standardize_features(
        X_train_agg, X_val_agg, X_test_agg
    )

    # Downsample training set to 1:1 healthy:failed
    print("\nDownsampling training set to 1:1 healthy:failed...")
    X_train_bal, y_train_bal = downsample_train(
        X_train_std,
        y_train,
        random_state=args.seed,
    )
    print(f"Balanced train: X={X_train_bal.shape}, y={y_train_bal.shape}")

    # Balanced validation subset (for threshold tuning)
    idx_bal_val = balanced_indices(y_val, random_state=args.seed)
    X_val_bal = X_val_std[idx_bal_val]
    y_val_bal = y_val[idx_bal_val]
    print(
        "\nBalanced validation subset:\n"
        f"  #failed windows (label=1): {np.sum(y_val_bal == 1)}\n"
        f"  #healthy windows (label=0): {np.sum(y_val_bal == 0)}\n"
        f"  Total windows:              {len(y_val_bal)}"
    )

    # Balanced test subset (for reporting, same as your other baselines)
    idx_bal_test = balanced_indices(y_test, random_state=args.seed)
    X_test_bal = X_test_std[idx_bal_test]
    y_test_bal = y_test[idx_bal_test]
    print(
        "\nBalanced test subset:\n"
        f"  #failed windows (label=1): {np.sum(y_test_bal == 1)}\n"
        f"  #healthy windows (label=0): {np.sum(y_test_bal == 0)}\n"
        f"  Total windows:              {len(y_test_bal)}"
    )

    # --------------------------------------------------------------
    # LightGBM model (if available)
    # --------------------------------------------------------------
    if HAS_LGBM:
        print("\n================ LightGBM baseline ================")
        lgbm = LGBMClassifier(
            n_estimators=args.lgbm_trees,
            objective="binary",
            boosting_type="gbdt",
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=args.seed,
            n_jobs=-1,
        )
        lgbm.fit(X_train_bal, y_train_bal)

        val_proba_lgbm = lgbm.predict_proba(X_val_std)[:, 1]
        val_bal_proba_lgbm = lgbm.predict_proba(X_val_bal)[:, 1]
        test_bal_proba_lgbm = lgbm.predict_proba(X_test_bal)[:, 1]

        print("\n[LightGBM] Test prob stats (balanced): "
              f"min={float(test_bal_proba_lgbm.min()):.4f}, "
              f"max={float(test_bal_proba_lgbm.max()):.4f}, "
              f"mean={float(test_bal_proba_lgbm.mean()):.4f}")

        # Choose threshold on balanced validation to maximize precision
        lgbm_thr, _, _, _ = find_best_threshold_for_precision(
            y_val_bal,
            val_bal_proba_lgbm,
            name="LightGBM",
        )
        print(f"\n[LightGBM] Using decision threshold {lgbm_thr:.3f}.")

        # Validation (unbalanced)
        evaluate_split(
            "LightGBM - Validation (unbalanced)",
            y_val,
            val_proba_lgbm,
            threshold=lgbm_thr,
        )

        # Test (balanced)
        evaluate_split(
            "LightGBM - Test (balanced 1:1)",
            y_test_bal,
            test_bal_proba_lgbm,
            threshold=lgbm_thr,
        )
    else:
        print("\n[Warning] lightgbm is not installed; skipping LightGBM model.")

    # --------------------------------------------------------------
    # Random Forest model
    # --------------------------------------------------------------
    print("\n================ Random Forest baseline ================")
    rf = RandomForestClassifier(
        n_estimators=args.rf_trees,
        max_depth=args.rf_max_depth,
        n_jobs=-1,
        random_state=args.seed,
        class_weight=None,  # train already balanced
    )
    rf.fit(X_train_bal, y_train_bal)

    val_proba_rf = rf.predict_proba(X_val_std)[:, 1]
    val_bal_proba_rf = rf.predict_proba(X_val_bal)[:, 1]
    test_bal_proba_rf = rf.predict_proba(X_test_bal)[:, 1]

    print("\n[RF] Test prob stats (balanced): "
          f"min={float(test_bal_proba_rf.min()):.4f}, "
          f"max={float(test_bal_proba_rf.max()):.4f}, "
          f"mean={float(test_bal_proba_rf.mean()):.4f}")

    rf_thr, _, _, _ = find_best_threshold_for_precision(
        y_val_bal,
        val_bal_proba_rf,
        name="RF",
    )
    print(f"\n[RF] Using decision threshold {rf_thr:.3f}.")

    evaluate_split(
        "RF - Validation (unbalanced)",
        y_val,
        val_proba_rf,
        threshold=rf_thr,
    )

    evaluate_split(
        "RF - Test (balanced 1:1)",
        y_test_bal,
        test_bal_proba_rf,
        threshold=rf_thr,
    )


if __name__ == "__main__":
    main()
