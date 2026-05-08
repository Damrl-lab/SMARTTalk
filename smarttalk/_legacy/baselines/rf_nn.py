"""
RF + NN baselines for binary SSD failure prediction on 30-day windows.

- Input: train.npz / val.npz / test.npz from n_day_window.py
  Each .npz must contain:
      X: float32, shape [N, T, F]  (e.g., T=30 days, F SMART attributes)
      y: int64,   shape [N]        (0 = healthy, 1 = failed)
      ttf: int32, shape [N]        (unused here)
      features:   shape [F]        (attribute names)

- Features:
    * We flatten each 30-day window into a single vector:
        [T, F] -> [T * F]
      Then standardize using mean/std from the training set.

- Models:
    * Random Forest (scikit-learn RandomForestClassifier)
    * Neural Network (scikit-learn MLPClassifier)

- Training:
    * Train on a **downsampled 1:1** balanced subset of the training data
      (same #healthy as #failed windows).

- Evaluation:
    * Validation: full (unbalanced) validation set.
    * Test: ratio-based subset using all failed windows plus a random sample
      of healthy windows.
    * For each model, print precision, recall, and F1 for the
      failed class (label=1), separately for val and test.

Usage example:
    python rf_nn_baseline_binary.py --data_dir processed/MB2_round1
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
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
# Config
# ----------------------------------------------------------------------


@dataclass
class BaselineConfig:
    n_trees: int = 400
    rf_max_depth: int | None = None
    nn_hidden: tuple = (256, 128)
    nn_max_iter: int = 50
    nn_batch_size: int = 256
    nn_lr: float = 1e-3
    seed: int = 0


# ----------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------


def flatten_windows(X: np.ndarray) -> np.ndarray:
    """
    Flatten each [T, F] window into a [T*F] feature vector.

    X: [N, T, F] -> [N, T*F]
    """
    N, T, F = X.shape
    X_flat = X.reshape(N, T * F)
    X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)
    return X_flat.astype(np.float32)


def standardize_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[StandardScaler, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a StandardScaler on the training features and transform
    train/val/test matrices.
    """
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X_train)

    X_train_std = scaler.transform(X_train).astype(np.float32)
    X_val_std = scaler.transform(X_val).astype(np.float32)
    X_test_std = scaler.transform(X_test).astype(np.float32)

    return scaler, X_train_std, X_val_std, X_test_std


# ----------------------------------------------------------------------
# Sampling helpers
# ----------------------------------------------------------------------


def downsample_train(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsample majority class to get ~1:1 ratio (healthy vs failed)
    for training.

    Returns balanced X_ds, y_ds.
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
    Build indices for a ratio-based subset of a split:
      - use ALL failed samples
      - randomly sample `healthy_per_fail` times as many healthy samples
        whenever enough healthy windows are available.

    This is used for the imbalanced test evaluation.
    """
    rng = check_random_state(random_state)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos == 0:
        raise ValueError("No positive (failed) samples in this split.")
    if n_neg == 0:
        raise ValueError("No negative (healthy) samples in this split.")

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
# Evaluation
# ----------------------------------------------------------------------
def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    name: str = "",
) -> tuple[float, float, float, float]:
    """
    Search over thresholds in [0, 1] to maximize F1 for the failed class (label=1).
    Returns (best_threshold, precision, recall, f1).
    """
    best_thr = 0.5
    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0

    # You can refine the grid if needed
    for thr in np.linspace(0.05, 0.95, 19):
        y_pred = (y_proba >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            pos_label=1,
            zero_division=0,
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_prec = prec
            best_rec = rec

    print(f"\n[{name}] Best threshold search on validation (balanced):")
    print(f"  best_thr={best_thr:.3f}, precision={best_prec:.4f}, "
          f"recall={best_rec:.4f}, f1={best_f1:.4f}")

    return best_thr, best_prec, best_rec, best_f1


def evaluate_split(
    name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute and print precision / recall / F1 for the failed class (label=1).
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
# Main training / evaluation logic
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Random Forest & Neural Network baselines on 30-day SSD windows",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="processed/MB2_round1",
        help="Directory with train.npz / val.npz / test.npz",
    )
    parser.add_argument(
        "--n_trees",
        type=int,
        default=400,
        help="Number of trees for RandomForest",
    )
    parser.add_argument(
        "--rf_max_depth",
        type=int,
        default=None,
        help="Max depth for RandomForest (None = unlimited)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--nn_hidden",
        type=int,
        nargs="+",
        default=[256, 128],
        help="Hidden layer sizes for MLPClassifier, e.g., --nn_hidden 256 128",
    )
    parser.add_argument(
        "--nn_max_iter",
        type=int,
        default=50,
        help="Max iterations for MLPClassifier",
    )
    parser.add_argument(
        "--nn_batch_size",
        type=int,
        default=256,
        help="Batch size for MLPClassifier",
    )
    parser.add_argument(
        "--nn_lr",
        type=float,
        default=1e-3,
        help="Learning rate for MLPClassifier (learning_rate_init)",
    )

    args = parser.parse_args()

    cfg = BaselineConfig(
        n_trees=args.n_trees,
        rf_max_depth=args.rf_max_depth,
        nn_hidden=tuple(args.nn_hidden),
        nn_max_iter=args.nn_max_iter,
        nn_batch_size=args.nn_batch_size,
        nn_lr=args.nn_lr,
        seed=args.seed,
    )

    np.random.seed(cfg.seed)

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

    # Flatten [T, F] windows into [T*F] feature vectors
    print("\nFlattening windows...")
    X_train_flat = flatten_windows(X_train)
    X_val_flat   = flatten_windows(X_val)
    X_test_flat  = flatten_windows(X_test)

    print("Standardizing features (fit on train, apply to val/test)...")
    scaler, X_train_std, X_val_std, X_test_std = standardize_features(
        X_train_flat, X_val_flat, X_test_flat
    )

    # Downsample training set to 1:1 healthy:failed
    X_train_bal, y_train_bal = downsample_train(
        X_train_std,
        y_train,
        random_state=cfg.seed,
    )

    print("\nBalanced training set:")
    print(f"  total windows: {len(y_train_bal)}")
    print(f"  #failed (1):   {np.sum(y_train_bal == 1)}")
    print(f"  #healthy (0):  {np.sum(y_train_bal == 0)}")

    # Build balanced test subset indices
    idx_bal_test = balanced_indices(y_test, random_state=cfg.seed)
    print("\nBalanced test subset (by windows):")
    print(f"  #failed (1):   {np.sum(y_test[idx_bal_test] == 1)}")
    print(f"  #healthy (0):  {np.sum(y_test[idx_bal_test] == 0)}")
    print(f"  total windows: {len(idx_bal_test)}")

    X_test_bal = X_test_std[idx_bal_test]
    y_test_bal = y_test[idx_bal_test]

    # Build balanced validation subset (for threshold tuning)
    idx_bal_val = balanced_indices(y_val, random_state=cfg.seed)
    X_val_bal = X_val_std[idx_bal_val]
    y_val_bal = y_val[idx_bal_val]


    # ------------------------------------------------------------------
    # Random Forest baseline
    # ------------------------------------------------------------------
        # ------------------------------------------------------------------
    # Random Forest baseline
    # ------------------------------------------------------------------
    print("\n================ Random Forest baseline ================")
    rf = RandomForestClassifier(
        n_estimators=cfg.n_trees,
        max_depth=cfg.rf_max_depth,
        n_jobs=-1,
        random_state=cfg.seed,
        class_weight=None,   # training set already balanced
    )
    rf.fit(X_train_bal, y_train_bal)

    # Probabilities
    y_val_proba_rf = rf.predict_proba(X_val_std)[:, 1]
    y_val_bal_proba_rf = rf.predict_proba(X_val_bal)[:, 1]
    y_test_proba_rf = rf.predict_proba(X_test_bal)[:, 1]

    # Choose threshold on balanced validation set (maximize F1)
    rf_best_thr, _, _, _ = find_best_threshold(
        y_val_bal,
        y_val_bal_proba_rf,
        name="RF",
    )
    print(f"\n[RF] Using decision threshold {rf_best_thr:.3f} for reporting.")

    # Validation (unbalanced) – report with tuned threshold
    rf_val_metrics = evaluate_split(
        "RF - Validation (unbalanced)",
        y_val,
        y_val_proba_rf,
        threshold=rf_best_thr,
    )

    # Test (balanced) – same threshold
    rf_test_metrics = evaluate_split(
        "RF - Test (balanced 1:1)",
        y_test_bal,
        y_test_proba_rf,
        threshold=rf_best_thr,
    )



    # ------------------------------------------------------------------
    # Neural Network baseline (MLP)
    # ------------------------------------------------------------------
    print("\n================ Neural Network baseline ================")
    mlp = MLPClassifier(
        hidden_layer_sizes=cfg.nn_hidden,
        activation="relu",
        solver="adam",
        batch_size=cfg.nn_batch_size,
        learning_rate_init=cfg.nn_lr,
        max_iter=cfg.nn_max_iter,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=cfg.seed,
        verbose=False,
    )
    mlp.fit(X_train_bal, y_train_bal)

     # Validation (unbalanced)
    y_val_proba_nn = mlp.predict_proba(X_val_std)[:, 1]
    nn_val_metrics = evaluate_split("NN - Validation (unbalanced)", y_val, y_val_proba_nn)

    # Test (balanced)
    y_test_proba_nn = mlp.predict_proba(X_test_bal)[:, 1]
    nn_test_metrics = evaluate_split("NN - Test (balanced 1:1)", y_test_bal, y_test_proba_nn)

    # Save metrics to file
    metrics_path = "baselines/rf_nn_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("Random Forest - Validation (unbalanced):\n")
        f.write(
            f"precision={rf_val_metrics['precision']:.4f}, "
            f"recall={rf_val_metrics['recall']:.4f}, "
            f"f1={rf_val_metrics['f1']:.4f}\n\n"
        )
        f.write("Random Forest - Test (balanced 1:1):\n")
        f.write(
            f"precision={rf_test_metrics['precision']:.4f}, "
            f"recall={rf_test_metrics['recall']:.4f}, "
            f"f1={rf_test_metrics['f1']:.4f}\n\n"
        )
        f.write("Neural Network - Validation (unbalanced):\n")
        f.write(
            f"precision={nn_val_metrics['precision']:.4f}, "
            f"recall={nn_val_metrics['recall']:.4f}, "
            f"f1={nn_val_metrics['f1']:.4f}\n\n"
        )
        f.write("Neural Network - Test (balanced 1:1):\n")
        f.write(
            f"precision={nn_test_metrics['precision']:.4f}, "
            f"recall={nn_test_metrics['recall']:.4f}, "
            f"f1={nn_test_metrics['f1']:.4f}\n"
        )



if __name__ == "__main__":
    main()
