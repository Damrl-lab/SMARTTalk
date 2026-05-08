"""
MVTRF-style binary SSD failure prediction baseline on 30-day windows.

This script:
  - Loads train.npz / val.npz / test.npz produced by n_day_window.py
  - Builds MVTRF-style multi-view features:
      * Raw features: last day's Telemetry log
      * Histogram features: per-attribute histograms over the 30-day window
      * Sequence features: CVAR / kurtosis / slope over G segments
  - Trains 4 RandomForest groups (raw, hist, seq, combined) as in MVTRF:
      each group sees a different feature view; final prediction is the
      average vote across the 4 groups.
  - Performs binary classification: 0 = healthy, 1 = failed
  - Evaluates precision / recall / F1 on:
      * Validation: full (unbalanced)
      * Test: ratio-based subset (all failed windows plus sampled healthy windows)

Assumed .npz format (per split):
    X: float32, shape [N, T, F]  (T=30 days, F SMART attributes)
    y: int64,   shape [N]        (0 = healthy, 1 = failed)
    ttf: int32, shape [N]        (unused here)
    features:   shape [F]        (attribute names)
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.utils import check_random_state


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------


def load_split(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one split (train / val / test) from an .npz file."""
    data = np.load(npz_path)
    X = data["X"]          # [N, T, F]
    y = data["y"]          # [N]
    ttf = data["ttf"]      # [N]
    features = data["features"]  # [F]
    return X, y, ttf, features


# ----------------------------------------------------------------------
# MVTRF-style feature extraction
# ----------------------------------------------------------------------


@dataclass
class MVTRFConfig:
    n_hist_bins: int = 50         # M in the paper (they use 100 by default)
    n_seq_segments: int = 4       # G in the paper (default 4)
    n_trees_total: int = 400      # total #trees across all views
    max_depth: int | None = None  # RF max depth (None = unlimited)
    random_state: int = 0         # base random seed


def compute_min_max_from_train(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-attribute min/max over the training set.

    X_train: [N, T, F]
    Returns:
        mins: [F]
        maxs: [F]
    """
    # Use nan-safe reductions in case some values are NaN
    mins = np.nanmin(X_train, axis=(0, 1))
    maxs = np.nanmax(X_train, axis=(0, 1))
    # Handle attributes that are all NaN
    all_nan = ~np.isfinite(mins) | ~np.isfinite(maxs)
    mins[all_nan] = 0.0
    maxs[all_nan] = 1.0
    # Avoid zero-width ranges
    same = mins == maxs
    maxs[same] = mins[same] + 1.0
    return mins.astype(np.float32), maxs.astype(np.float32)


def compute_histogram_edges(
    mins: np.ndarray,
    maxs: np.ndarray,
    n_hist_bins: int,
) -> np.ndarray:
    """
    Compute per-attribute histogram bin edges.

    We approximate the paper's bucket definition by using
    equally spaced bins between per-attribute min and max.

    Args:
        mins: [F] per-attribute minimums from training data
        maxs: [F] per-attribute maximums from training data
        n_hist_bins: number of histogram bins M

    Returns:
        edges: [F, n_hist_bins + 1]
    """
    F = mins.shape[0]
    edges = np.empty((F, n_hist_bins + 1), dtype=np.float32)
    for f in range(F):
        vmin = float(mins[f])
        vmax = float(maxs[f])
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        edges[f] = np.linspace(vmin, vmax, n_hist_bins + 1, dtype=np.float32)
    return edges


def raw_features(X: np.ndarray) -> np.ndarray:
    """
    Raw features: the last Telemetry log (short-term snapshot).

    In MVTRF, raw features come from a single Telemetry vector D_T. :contentReference[oaicite:1]{index=1}

    X: [N, T, F] -> [N, F]
    """
    last = X[:, -1, :]  # [N, F]
    last = np.nan_to_num(last, nan=0.0, posinf=0.0, neginf=0.0)
    return last.astype(np.float32)


def histogram_features(
    X: np.ndarray,
    hist_edges: np.ndarray,
) -> np.ndarray:
    """
    Faster, vectorized histogram features over the full window.

    X: [N, T, F]
    hist_edges: [F, M+1]

    Returns:
        hist_feats: [N, F * M]
    """
    N, T, F = X.shape
    M = hist_edges.shape[1] - 1
    hist_feats = np.zeros((N, F, M), dtype=np.float32)

    for f in range(F):
        edges = hist_edges[f]         # [M+1]
        vals = X[:, :, f]             # [N, T]
        finite = np.isfinite(vals)    # [N, T]

        # Replace non-finite with a value below edges[0]
        v = vals.copy()
        v[~finite] = edges[0] - 1.0

        # Bin indices via searchsorted
        bin_idx = np.searchsorted(edges, v, side="right") - 1  # [N, T]
        # Clip high side; low side (<0) stays <0
        bin_idx[bin_idx >= M] = M - 1

        # Only count finite values with valid bins
        valid = finite & (bin_idx >= 0)

        # Flatten for np.add.at
        i_idx = np.repeat(np.arange(N), T)
        b_idx = bin_idx.reshape(-1)
        vmask = valid.reshape(-1)

        i_idx = i_idx[vmask]
        b_idx = b_idx[vmask]

        counts = np.zeros((N, M), dtype=np.float32)
        np.add.at(counts, (i_idx, b_idx), 1)

        # Normalize by number of finite time steps per sample
        n_finite = finite.sum(axis=1).astype(np.float32)  # [N]
        n_finite[n_finite == 0] = 1.0
        counts /= n_finite[:, None]

        hist_feats[:, f, :] = counts

    return hist_feats.reshape(N, F * M)



def sequence_features(
    X: np.ndarray,
    n_segments: int,
) -> np.ndarray:
    """
    Sequence-related features: CVAR, kurtosis, slope over G segments.

    Following MVTRF, we divide the long-term data DT-L..DT into G
    equal time segments, and compute: coefficient of variation,
    kurtosis, and slope per segment and per attribute. :contentReference[oaicite:3]{index=3}

    Here long-term == 30 days (our window), so L = T.

    Args:
        X: [N, T, F]
        n_segments: G

    Returns:
        seq_feats: [N, F * G * 3]  (CVAR, KURT, SLOPE for each segment)
    """
    N, T, F = X.shape
    seg_feat_list = []

    # Replace non-finite with NaN to use nan-aware stats
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, np.nan)

    for g in range(n_segments):
        start = (g * T) // n_segments
        end = ((g + 1) * T) // n_segments
        if end <= start:
            # degenerate, skip
            continue

        seg = X[:, start:end, :]  # [N, Lg, F]
        Lg = end - start

        # mean over time
        mean = np.nanmean(seg, axis=1)  # [N, F]
        # center
        diff = seg - mean[:, None, :]   # [N, Lg, F]
        diff2 = diff ** 2
        diff4 = diff ** 4

        # second and fourth moments (nan-aware)
        m2 = np.nanmean(diff2, axis=1)  # [N, F]
        m4 = np.nanmean(diff4, axis=1)  # [N, F]
        std = np.sqrt(m2)

        # coefficient of variation: std / mean
        cvar = np.zeros_like(std)
        valid_mean = np.abs(mean) > 1e-8
        cvar[valid_mean] = std[valid_mean] / mean[valid_mean]

        # kurtosis: m4 / m2^2 - 3
        kurt = np.zeros_like(m4)
        valid_var = m2 > 1e-8
        kurt[valid_var] = m4[valid_var] / (m2[valid_var] ** 2) - 3.0

        # slope: (last - first) / (te - ts)
        first = seg[:, 0, :]   # [N, F]
        last = seg[:, -1, :]   # [N, F]
        denom = max(Lg - 1, 1)
        slope = (last - first) / float(denom)

        # Replace NaNs/Infs with zeros
        cvar = np.nan_to_num(cvar, nan=0.0, posinf=0.0, neginf=0.0)
        kurt = np.nan_to_num(kurt, nan=0.0, posinf=0.0, neginf=0.0)
        slope = np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0)

        # [N, F, 3] -> [N, F*3]
        seg_feats = np.stack([cvar, kurt, slope], axis=2)  # [N, F, 3]
        seg_feats = seg_feats.reshape(N, F * 3)
        seg_feat_list.append(seg_feats)

    if not seg_feat_list:
        # fallback: all zeros
        return np.zeros((N, F * n_segments * 3), dtype=np.float32)

    seq_feats = np.concatenate(seg_feat_list, axis=1)  # [N, F * G * 3]
    return seq_feats.astype(np.float32)


def extract_mvtrf_views(
    X: np.ndarray,
    config: MVTRFConfig,
    hist_edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract all four feature views for MVTRF:

      - raw_feats:   [N, F]
      - hist_feats:  [N, F * M]
      - seq_feats:   [N, F * G * 3]
      - comb_feats:  concatenation of the above three

    These correspond to the four decision-tree groups in MVTRF:
      raw / histogram / sequence / combined. :contentReference[oaicite:4]{index=4}
    """
    raw_feats = raw_features(X)
    hist_feats = histogram_features(X, hist_edges)
    seq_feats = sequence_features(X, config.n_seq_segments)

    comb_feats = np.concatenate([raw_feats, hist_feats, seq_feats], axis=1)

    # Ensure no NaNs/Infs
    raw_feats = np.nan_to_num(raw_feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    hist_feats = np.nan_to_num(hist_feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    seq_feats = np.nan_to_num(seq_feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    comb_feats = np.nan_to_num(comb_feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return raw_feats, hist_feats, seq_feats, comb_feats


# ----------------------------------------------------------------------
# MVTRF: multi-view Random Forest
# ----------------------------------------------------------------------


def train_mvtrf_forests(
    X_raw: np.ndarray,
    X_hist: np.ndarray,
    X_seq: np.ndarray,
    X_comb: np.ndarray,
    y: np.ndarray,
    config: MVTRFConfig,
) -> Dict[str, RandomForestClassifier]:
    """
    Train four RandomForestClassifier groups as in MVTRF:

      - 'raw'  : trees trained on raw features only
      - 'hist' : trees trained on histogram features only
      - 'seq'  : trees trained on sequence features only
      - 'comb' : trees trained on combined features

    All groups use the same #trees so that averaging their
    probability predictions matches "equal voting" across
    all decision trees. :contentReference[oaicite:5]{index=5}
    """
    n_each = max(1, config.n_trees_total // 4)

    def make_rf(seed: int) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=n_each,
            max_depth=config.max_depth,
            n_jobs=-1,
            class_weight="balanced",
            random_state=seed,
        )

    forests: Dict[str, RandomForestClassifier] = {}

    print("\nTraining MVTRF forests (multi-view Random Forest)...")
    print(f"  total trees: {config.n_trees_total}  ->  {n_each} per view")

    forests["raw"] = make_rf(config.random_state + 0).fit(X_raw, y)
    forests["hist"] = make_rf(config.random_state + 1).fit(X_hist, y)
    forests["seq"] = make_rf(config.random_state + 2).fit(X_seq, y)
    forests["comb"] = make_rf(config.random_state + 3).fit(X_comb, y)

    return forests


def mvtrf_predict_proba(
    forests: Dict[str, RandomForestClassifier],
    X_raw: np.ndarray,
    X_hist: np.ndarray,
    X_seq: np.ndarray,
    X_comb: np.ndarray,
) -> np.ndarray:
    """
    Aggregate predictions from the four MVTRF views.

    We average the predicted probability p(y=1) from each
    view's forest. With equal #trees per forest, this is
    equivalent to equal-weight voting over all trees.
    """
    p_raw = forests["raw"].predict_proba(X_raw)[:, 1]
    p_hist = forests["hist"].predict_proba(X_hist)[:, 1]
    p_seq = forests["seq"].predict_proba(X_seq)[:, 1]
    p_comb = forests["comb"].predict_proba(X_comb)[:, 1]

    proba = (p_raw + p_hist + p_seq + p_comb) / 4.0
    return proba


# ----------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------

def find_best_threshold_for_precision(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    name: str = "",
) -> tuple[float, float, float, float]:
    """
    Search over thresholds to get high precision and low recall for the failed class (label=1),
    similar to the operating point reported in MVTRF.

    Strategy:
      - Scan thresholds in [0.05, 0.95].
      - Among thresholds with recall > 0, pick the one with MAX precision.
      - If all have recall == 0, fall back to the best F1.
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

    # 1) thresholds with non-zero recall
    non_zero = [c for c in candidates if c[2] > 0.0]
    if non_zero:
        # pick the one with highest precision
        thr, prec, rec, f1 = max(non_zero, key=lambda x: x[1])
        best_thr, best_prec, best_rec, best_f1 = thr, prec, rec, f1
    else:
        # fall back to best F1 (all recall=0)
        thr, prec, rec, f1 = max(candidates, key=lambda x: x[3])
        best_thr, best_prec, best_rec, best_f1 = thr, prec, rec, f1

    print(f"\n[{name}] Threshold search on validation (ratio-based subset):")
    print(f"  best_thr={best_thr:.3f}, precision={best_prec:.4f}, "
          f"recall={best_rec:.4f}, f1={best_f1:.4f}")

    return best_thr, best_prec, best_rec, best_f1



def evaluate_split(
    name: str,
    proba: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate precision / recall / F1 for the failed class (label=1)
    at the given decision threshold.
    """
    y_pred = (proba >= threshold).astype(int)


    prec, rec, f1, _ = precision_recall_fscore_support(
        y, y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    print(f"\n=== {name} ===")
    print(f"Samples: {len(y)}")
    print(f"Precision (fail=1): {prec:.4f}")
    print(f"Recall    (fail=1): {rec:.4f}")
    print(f"F1        (fail=1): {f1:.4f}")
    print("\nClassification report (0=healthy, 1=failed):")
    print(classification_report(y, y_pred, digits=4, zero_division=0))

    return {"precision": float(prec), "recall": float(rec), "f1": float(f1)}


def balanced_indices(
    y: np.ndarray,
    random_state: int = 0,
    healthy_per_fail: float = 1.0,
) -> np.ndarray:
    """
    Build indices for a ratio-based subset:
      - use ALL failed windows
      - randomly sample `healthy_per_fail` times as many healthy windows.
    """
    rng = check_random_state(random_state)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos == 0:
        raise ValueError("No positive (failed) samples in test set.")
    if n_neg == 0:
        raise ValueError("No negative (healthy) samples in test set.")

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


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="MVTRF-style binary SSD failure baseline on 30-day windows",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="processed/MB2_round3",
        help="Directory containing train.npz / val.npz / test.npz",
    )
    parser.add_argument(
        "--n_hist_bins",
        type=int,
        default=50,  # paper uses 100; 50 is a lighter but similar choice
        help="Number of histogram bins per attribute (M in MVTRF)",
    )
    parser.add_argument(
        "--n_seq_segments",
        type=int,
        default=4,   # G in MVTRF
        help="Number of time segments for sequence features (G in MVTRF)",
    )
    parser.add_argument(
        "--n_trees_total",
        type=int,
        default=400,
        help="Total number of trees across all 4 MVTRF views",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Max depth for each RandomForest tree (None = unlimited)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )

    args = parser.parse_args()
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

    T = X_train.shape[1]
    print(f"\nWindow length T: {T} days (should be 30 for SMARTTalk setup)")

    config = MVTRFConfig(
        n_hist_bins=args.n_hist_bins,
        n_seq_segments=args.n_seq_segments,
        n_trees_total=args.n_trees_total,
        max_depth=args.max_depth,
        random_state=args.seed,
    )

    # 1) Histogram bin edges from training data (multi-view long-term stats)
    print("\nComputing histogram bin edges from training data...")
    mins, maxs = compute_min_max_from_train(X_train)
    hist_edges = compute_histogram_edges(mins, maxs, config.n_hist_bins)

    # 2) Extract MVTRF views for each split
    print("Extracting MVTRF feature views (raw / hist / seq / combined)...")
    tr_raw, tr_hist, tr_seq, tr_comb = extract_mvtrf_views(X_train, config, hist_edges)
    val_raw, val_hist, val_seq, val_comb = extract_mvtrf_views(X_val, config, hist_edges)
    te_raw, te_hist, te_seq, te_comb = extract_mvtrf_views(X_test, config, hist_edges)

    print("\nFeature matrix shapes (samples × features):")
    print(f"  train raw : {tr_raw.shape}")
    print(f"  train hist: {tr_hist.shape}")
    print(f"  train seq : {tr_seq.shape}")
    print(f"  train comb: {tr_comb.shape}")

    # 3) Train MVTRF forests (multi-view RandomForest)
    forests = train_mvtrf_forests(
        tr_raw, tr_hist, tr_seq, tr_comb,
        y_train,
        config,
    )

    # 4) Validation: compute probabilities
    val_proba = mvtrf_predict_proba(forests, val_raw, val_hist, val_seq, val_comb)

    # Build a balanced validation subset to choose the operating threshold
    idx_bal_val = balanced_indices(y_val, random_state=args.seed)
    y_val_bal = y_val[idx_bal_val]
    val_proba_bal = val_proba[idx_bal_val]

    # Choose a high-precision, low-recall threshold on balanced validation
    best_thr, _, _, _ = find_best_threshold_for_precision(
        y_val_bal,
        val_proba_bal,
        name="MVTRF",
    )
    print(f"\n[MVTRF] Using decision threshold {best_thr:.3f} for reporting.\n")

    # Validation (full, unbalanced) at tuned threshold
    evaluate_split("Validation (unbalanced)", val_proba, y_val, threshold=best_thr)


    # 5) Balanced test subset and evaluation
    idx_bal = balanced_indices(y_test, random_state=args.seed)
    print("\nBalanced test subset:")
    print(f"  #failed windows (label=1):  {np.sum(y_test[idx_bal] == 1)}")
    print(f"  #healthy windows (label=0): {np.sum(y_test[idx_bal] == 0)}")
    print(f"  Total windows:              {len(idx_bal)}")

    te_raw_bal  = te_raw[idx_bal]
    te_hist_bal = te_hist[idx_bal]
    te_seq_bal  = te_seq[idx_bal]
    te_comb_bal = te_comb[idx_bal]
    y_test_bal  = y_test[idx_bal]

    te_proba_bal = mvtrf_predict_proba(forests, te_raw_bal, te_hist_bal, te_seq_bal, te_comb_bal)
    # Optionally: sanity-check probabilities
    print("\n[MVTRF] Test prob stats (ratio-based subset): "
          f"min={float(te_proba_bal.min()):.4f}, "
          f"max={float(te_proba_bal.max()):.4f}, "
          f"mean={float(te_proba_bal.mean()):.4f}")

    evaluate_split(
        "Test (ratio-based healthy:failed subset)",
        te_proba_bal,
        y_test_bal,
        threshold=best_thr,
    )



if __name__ == "__main__":
    main()
