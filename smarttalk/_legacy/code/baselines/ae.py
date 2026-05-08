#!/usr/bin/env python3
"""
Autoencoder (AE) baseline for SSD failure prediction (binary, 0=healthy, 1=failed).

Based on the 1-class autoencoder idea from:
  C. Chakraborttii and H. Litz, "Improving the Accuracy, Adaptability, and
  Interpretability of SSD Failure Prediction Models", SoCC'20. 1-class AE
  is trained ONLY on healthy observations and uses reconstruction error
  as an anomaly score.  (We adapt it to 30-day windows.)

Assumed .npz format per split (train/val/test):
    X: float32, shape [N, T, F]  (e.g., T=30 days, F SMART attributes)
    y: int64,   shape [N]        (0 = healthy, 1 = failed)
    ttf: int32, shape [N]        (unused here)
    features:   shape [F]        (attribute names)

Pipeline:
  1) Flatten windows: [T, F] -> [T*F]; min-max normalize using train stats.
  2) TRAIN Autoencoder on healthy TRAIN windows only (1-class).
  3) Compute reconstruction error per window on train/val/test.
  4) Build balanced val and test subsets (equal #healthy and #failed).
  5) Pick threshold on balanced val that maximizes F1 (failed=1).
  6) Evaluate on balanced val & balanced test: precision, recall, F1.

Usage example:
    python ae_baseline_binary.py --data_dir processed/MB2_round1
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    precision_recall_fscore_support,
    precision_recall_curve,
    classification_report,
)
from sklearn.utils import check_random_state


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------


def load_split(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    X = data["X"]          # [N, T, F]
    y = data["y"]          # [N]
    ttf = data["ttf"]      # [N]
    features = data["features"]  # [F]
    return X, y, ttf, features


# ----------------------------------------------------------------------
# Config / helpers
# ----------------------------------------------------------------------


@dataclass
class AEConfig:
    hidden_dims: Tuple[int, int] = (512, 128)
    latent_dim: int = 32
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 50
    seed: int = 0


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten_windows(X: np.ndarray) -> np.ndarray:
    """
    Flatten each [T, F] window into [T*F] vector.
    X: [N, T, F] -> [N, T*F]
    """
    N, T, F = X.shape
    X_flat = X.reshape(N, T * F)
    X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)
    return X_flat.astype(np.float32)


def minmax_fit_transform(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit per-dimension min/max on X_train and return:
      - X_train_scaled
      - mins, ranges
    Scaling: x_scaled = (x - mins) / ranges, ranges>=1e-6
    """
    mins = np.nanmin(X_train, axis=0)
    maxs = np.nanmax(X_train, axis=0)

    # Handle NaNs and zero ranges
    mins = np.where(np.isfinite(mins), mins, 0.0)
    maxs = np.where(np.isfinite(maxs), maxs, 1.0)
    ranges = maxs - mins
    ranges[ranges < 1e-6] = 1.0

    X_scaled = (X_train - mins) / ranges
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return X_scaled.astype(np.float32), mins.astype(np.float32), ranges.astype(np.float32)


def minmax_transform(X: np.ndarray, mins: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    X_scaled = (X - mins) / ranges
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return X_scaled.astype(np.float32)


def balanced_indices(y: np.ndarray, random_state: int = 0, healthy_per_fail: float = 1.0) -> np.ndarray:
    """
    Build indices for a ratio-based subset:
      - use ALL failed (y=1)
      - randomly sample `healthy_per_fail` times as many healthy (y=0).
    """
    rng = check_random_state(random_state)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    if n_pos == 0:
        raise ValueError("No positive (failed) samples in split.")
    if n_neg == 0:
        raise ValueError("No negative (healthy) samples in split.")

    target_neg = max(1, int(round(n_pos * healthy_per_fail)))
    if n_neg >= target_neg:
        pos_sel = pos_idx
        neg_sel = rng.choice(neg_idx, size=target_neg, replace=False)
    else:
        neg_sel = neg_idx
        max_pos = max(1, int(np.floor(n_neg / healthy_per_fail)))
        pos_sel = pos_idx if max_pos >= n_pos else rng.choice(pos_idx, size=max_pos, replace=False)

    idx = np.concatenate([pos_sel, neg_sel])
    rng.shuffle(idx)
    return idx


# ----------------------------------------------------------------------
# Dataset / model
# ----------------------------------------------------------------------


class AETensorDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]
        return x, x  # input == target


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(512, 128), latent_dim=32):
        super().__init__()
        h1, h2 = hidden_dims

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, latent_dim),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, input_dim),
            nn.Sigmoid(),  # inputs scaled to [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out


# ----------------------------------------------------------------------
# Train / inference
# ----------------------------------------------------------------------


def train_autoencoder(
    model: Autoencoder,
    X_train_healthy: np.ndarray,
    cfg: AEConfig,
    device: torch.device,
) -> None:
    """
    Train AE on healthy training windows only (1-class).
    """
    dataset = AETensorDataset(X_train_healthy)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss(reduction="mean")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for x_in, x_target in loader:
            x_in = x_in.to(device)
            x_target = x_target.to(device)

            optimizer.zero_grad()
            x_hat = model(x_in)
            loss = loss_fn(x_hat, x_target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(batches, 1)
        print(f"Epoch {epoch:03d} / {cfg.epochs}: AE train loss = {avg_loss:.6f}")


@torch.no_grad()
def compute_reconstruction_errors(
    model: Autoencoder,
    X: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Compute per-sample reconstruction MSE.
    """
    model.eval()
    dataset = AETensorDataset(X)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, drop_last=False)

    errs = []
    loss_fn = nn.MSELoss(reduction="none")

    for x_in, x_target in loader:
        x_in = x_in.to(device)
        x_target = x_target.to(device)
        x_hat = model(x_in)
        # loss per sample, per feature
        l = loss_fn(x_hat, x_target)  # [B, D]
        # mean over features -> [B]
        l = l.mean(dim=1).cpu().numpy()
        errs.append(l)

    if len(errs) == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(errs, axis=0).astype(np.float32)


def choose_threshold_on_val(
    errors_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """
    Pick reconstruction-error threshold that maximizes F1 on validation data.
    Higher error => more likely failed.

    We use precision_recall_curve to scan thresholds.
    """
    # scores = errors; label 1 = failed
    precision, recall, thresholds = precision_recall_curve(y_val, errors_val, pos_label=1)

    # thresholds has len = len(precision) - 1
    best_f1 = -1.0
    best_thr = None

    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if p + r == 0:
            continue
        f1 = 2 * p * r / (p + r)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = t

    if best_thr is None:
        # Fallback: median error
        best_thr = float(np.median(errors_val))
        best_f1 = 0.0

    print(f"\nChosen threshold on validation (balanced): {best_thr:.6f} (F1={best_f1:.4f})")
    return float(best_thr)


def evaluate_split(
    name: str,
    errors: np.ndarray,
    y: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Classify with given threshold and compute precision/recall/F1 for failed=1.
    """
    y_pred = (errors >= threshold).astype(int)

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


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Autoencoder (1-class) SSD failure baseline on 30-day windows",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="processed/MB2_round1",
        help="Directory with train.npz / val.npz / test.npz",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs for AE",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for AE training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for AE",
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs=2,
        default=[512, 128],
        help="Two hidden layer sizes for AE encoder/decoder, e.g. --hidden_dims 512 128",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=32,
        help="Latent dimension for AE",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )

    args = parser.parse_args()

    cfg = AEConfig(
        hidden_dims=(args.hidden_dims[0], args.hidden_dims[1]),
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
    )

    set_seed(cfg.seed)

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

    # Flatten and scale
    print("\nFlattening windows and computing min-max scaling from train...")
    X_train_flat = flatten_windows(X_train)
    X_val_flat   = flatten_windows(X_val)
    X_test_flat  = flatten_windows(X_test)

    X_train_scaled, mins, ranges = minmax_fit_transform(X_train_flat)
    X_val_scaled   = minmax_transform(X_val_flat, mins, ranges)
    X_test_scaled  = minmax_transform(X_test_flat, mins, ranges)

    N_train, D = X_train_scaled.shape
    print(f"\nFlattened feature dimension: D = {D}")

    # 1-class training: use ONLY healthy windows from training set
    healthy_mask = (y_train == 0)
    if not np.any(healthy_mask):
        raise ValueError("No healthy windows (y=0) in training set for 1-class AE.")

    X_train_healthy = X_train_scaled[healthy_mask]
    print(f"\nHealthy training windows for AE: {X_train_healthy.shape[0]} / {N_train}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Build and train AE
    model = Autoencoder(
        input_dim=D,
        hidden_dims=cfg.hidden_dims,
        latent_dim=cfg.latent_dim,
    )
    train_autoencoder(model, X_train_healthy, cfg, device)

    # Compute reconstruction errors for full splits
    print("\nComputing reconstruction errors for train/val/test...")
    train_errors = compute_reconstruction_errors(model, X_train_scaled, device)
    val_errors   = compute_reconstruction_errors(model, X_val_scaled, device)
    test_errors  = compute_reconstruction_errors(model, X_test_scaled, device)

    print("\nError stats (mean, std) on healthy train samples:")
    print(f"  mean: {train_errors[healthy_mask].mean():.6f}")
    print(f"  std : {train_errors[healthy_mask].std():.6f}")

    # Build balanced val and test subsets (equal #healthy and #failed)
    idx_val_bal = balanced_indices(y_val, random_state=cfg.seed)
    idx_test_bal = balanced_indices(y_test, random_state=cfg.seed + 1)

    val_err_bal = val_errors[idx_val_bal]
    val_y_bal   = y_val[idx_val_bal]

    test_err_bal = test_errors[idx_test_bal]
    test_y_bal   = y_test[idx_test_bal]

    print("\nBalanced validation subset:")
    print(f"  #failed (1): {np.sum(val_y_bal == 1)}")
    print(f"  #healthy (0): {np.sum(val_y_bal == 0)}")
    print(f"  total: {len(val_y_bal)}")

    print("\nBalanced test subset:")
    print(f"  #failed (1): {np.sum(test_y_bal == 1)}")
    print(f"  #healthy (0): {np.sum(test_y_bal == 0)}")
    print(f"  total: {len(test_y_bal)}")

    # Choose threshold on balanced validation set
    thr = choose_threshold_on_val(val_err_bal, val_y_bal)

    # Evaluate on balanced validation & test
    evaluate_split("AE - Validation (balanced)", val_err_bal, val_y_bal, thr)
    evaluate_split("AE - Test (balanced)", test_err_bal, test_y_bal, thr)


if __name__ == "__main__":
    main()
