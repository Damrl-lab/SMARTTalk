"""
MSFRD-style baseline (dynamic mutation + similarity-based classification) for
binary SSD failure prediction on 30-day windows.

This script:
  - Loads train.npz / val.npz / test.npz produced by n_day_window.py
  - Trains a simple time-series forecaster on HEALTHY training windows only
    (0 = healthy) to learn normal trends, inspired by MSFRD's Informer-based
    mutation feature extraction. [Zhang et al., USENIX ATC'24]
  - Learns per-attribute rarity weights W[n] jointly with the forecaster using
    a weighted loss similar to Eq. (1) in the paper.
  - For each window, computes mutation features as prediction errors between
    predicted future and actual future telemetry values.
  - Uses a k-nearest neighbors classifier (in weighted mutation space) as a
    mutation-similarity classifier for binary labels (0=healthy, 1=failed).
  - Evaluates precision/recall/F1 for the failed class on:
      * Validation: full (unbalanced).
      * Test: ratio-based subset using all failed windows plus sampled healthy windows.

Differences from the full MSFRD:
  - We focus on binary classification instead of the 4-level failure rating.
  - We replace Informer with a compact MLP forecaster for practicality.
  - We keep the core ideas: dynamic prediction-error-based mutation features
    and similarity-based prediction over historical windows.

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
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_random_state

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


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
# Configs and models
# ----------------------------------------------------------------------


@dataclass
class MSFRDConfig:
    time_in: int = 20          # history length H (days) inside 30-day window
    time_out: int = 10         # prediction horizon F (days); time_in + time_out <= 30
    hidden_dim: int = 256      # MLP hidden size
    n_epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    k_neighbors: int = 5       # k for k-NN similarity
    random_state: int = 0


class SimpleForecaster(nn.Module):
    """
    Compact MLP forecaster:
      input:  [B, T_in, F]
      output: [B, T_out, F]
    """

    def __init__(self, time_in: int, time_out: int, num_features: int, hidden_dim: int):
        super().__init__()
        self.time_in = time_in
        self.time_out = time_out
        self.num_features = num_features

        in_dim = time_in * num_features
        out_dim = time_out * num_features

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T_in, F]
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        h = F.relu(self.fc1(x_flat))
        out = self.fc2(h)
        return out.view(B, self.time_out, self.num_features)


class MSFRDMutationModel(nn.Module):
    """
    Wrapper for forecaster + rarity weights.

    Rarity weights W[n] (per attribute) are learned jointly with the forecaster
    using a weighted loss similar to Eq. (1):
        loss = mean( (error^2 * W[n]) ) + mean(exp(-W[n]))
    """

    def __init__(self, time_in: int, time_out: int, num_features: int, hidden_dim: int):
        super().__init__()
        self.forecaster = SimpleForecaster(time_in, time_out, num_features, hidden_dim)
        # Trainable rarity logits per attribute (ensure W > 0 via softplus)
        self.rarity_logits = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forecaster(x)

    def rarity_weights(self) -> torch.Tensor:
        # Positive rarity weights W[n] = softplus(logit) + eps
        return F.softplus(self.rarity_logits) + 1e-3


# ----------------------------------------------------------------------
# Training mutation model on healthy windows
# ----------------------------------------------------------------------


def train_mutation_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: MSFRDConfig,
    device: torch.device,
) -> MSFRDMutationModel:
    """
    Train the MSFRD mutation model on HEALTHY training windows only (y=0).

    For each window [T=30, F], we use the first time_in days as history and
    the next time_out days as prediction target.
    """
    N, T, F_feat = X_train.shape
    assert config.time_in + config.time_out <= T, "time_in + time_out must be <= window length"

    # Select healthy windows
    healthy_mask = (y_train == 0)
    X_h = X_train[healthy_mask]
    if X_h.shape[0] == 0:
        raise ValueError("No healthy windows in training data to train mutation model.")

    X_h = np.nan_to_num(X_h, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    X_in = X_h[:, :config.time_in, :]                 # [N_h, T_in, F]
    X_out = X_h[:, config.time_in:config.time_in + config.time_out, :]  # [N_h, T_out, F]

    dataset = TensorDataset(
        torch.from_numpy(X_in),
        torch.from_numpy(X_out),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = MSFRDMutationModel(config.time_in, config.time_out, F_feat, config.hidden_dim)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(1, config.n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            pred = model(x_batch)       # [B, T_out, F]
            W = model.rarity_weights()  # [F]

            error = y_batch - pred      # [B, T_out, F]
            sq_err = error ** 2

            # Mean squared error over time
            sq_err_mean_t = sq_err.mean(dim=1)  # [B, F]

            # Broadcast rarity weights and compute weighted MSE
            W_b = W.unsqueeze(0)               # [1, F]
            weighted_mse = (sq_err_mean_t * W_b).mean()

            # Penalty term to avoid W -> 0
            penalty = torch.exp(-W).mean()

            loss = weighted_mse + penalty
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch:03d} / {config.n_epochs}: train loss = {avg_loss:.6f}")

    return model


# ----------------------------------------------------------------------
# Mutation feature extraction
# ----------------------------------------------------------------------


def extract_mutation_features(
    model: MSFRDMutationModel,
    X: np.ndarray,
    config: MSFRDConfig,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mutation features and rarity weights for a split.

    For each window, we:
      - Take first time_in days as history input.
      - Predict next time_out days.
      - Mutation = (actual_future - predicted_future) flattened to [F * time_out].

    Returns:
        mut_feats: [N, F * time_out] float32
        rarity:   [F] float32  (same for all splits)
    """
    model.eval()
    N, T, F_feat = X.shape
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    assert config.time_in + config.time_out <= T

    X_in = X[:, :config.time_in, :]                 # [N, T_in, F]
    X_out = X[:, config.time_in:config.time_in + config.time_out, :]  # [N, T_out, F]

    # Run in batches to avoid GPU/CPU memory issues
    batch_size = 1024
    mut_list = []

    with torch.no_grad():
        for i in range(0, N, batch_size):
            x_batch = X_in[i:i + batch_size]
            y_batch = X_out[i:i + batch_size]
            if x_batch.shape[0] == 0:
                continue

            x_t = torch.from_numpy(x_batch).to(device)
            y_t = torch.from_numpy(y_batch).to(device)

            pred = model(x_t)  # [B, T_out, F]
            err = (y_t - pred).cpu().numpy()       # [B, T_out, F]

            mut = err.reshape(err.shape[0], -1)    # [B, F * T_out]
            mut_list.append(mut.astype(np.float32))

    mut_feats = np.concatenate(mut_list, axis=0) if mut_list else np.zeros((N, F_feat * config.time_out), dtype=np.float32)
    rarity = model.rarity_weights().detach().cpu().numpy().astype(np.float32)  # [F]

    return mut_feats, rarity


# ----------------------------------------------------------------------
# Similarity-based classifier (k-NN)
# ----------------------------------------------------------------------


def train_similarity_classifier(
    mut_train: np.ndarray,
    y_train: np.ndarray,
    rarity: np.ndarray,
    config: MSFRDConfig,
) -> KNeighborsClassifier:
    """
    Train a k-NN classifier on mutation features.

    Weighted Euclidean distance with rarity weights W[n] can be implemented
    by scaling each feature dimension by sqrt(W[n]) before standard Euclidean.
    For F attributes and time_out steps, we expand W[n] across the horizon.
    """
    N_train, D = mut_train.shape
    F_feat = rarity.shape[0]
    assert D % F_feat == 0, "Mutation feature dim should be multiple of #features"

    steps = D // F_feat

    # Build per-dimension weights: repeat W[n] for each predicted time step
    W_expand = np.repeat(rarity, steps)             # [D]
    scale = np.sqrt(W_expand + 1e-8).astype(np.float32)

    mut_train_scaled = mut_train * scale[None, :]

    knn = KNeighborsClassifier(
        n_neighbors=config.k_neighbors,
        weights="distance",
        metric="euclidean",
        n_jobs=-1,
    )
    knn.fit(mut_train_scaled, y_train)
    return knn, scale


def predict_similarity_classifier(
    knn: KNeighborsClassifier,
    mut: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    mut_scaled = mut * scale[None, :]
    proba = knn.predict_proba(mut_scaled)[:, 1]  # p(y=1)
    return proba


# ----------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------


def evaluate_split(
    name: str,
    proba: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    y_pred = (proba >= 0.5).astype(int)

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
      - randomly sample `healthy_per_fail` times as many healthy windows
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
        description="MSFRD-style binary SSD failure baseline (mutation + similarity) on 30-day windows",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="processed/MB2_round3",
        help="Directory containing train.npz / val.npz / test.npz",
    )
    parser.add_argument(
        "--time_in",
        type=int,
        default=20,
        help="History length (days) inside each 30-day window",
    )
    parser.add_argument(
        "--time_out",
        type=int,
        default=10,
        help="Prediction horizon (days) inside each 30-day window",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for MLP forecaster",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs for mutation model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for mutation model training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for mutation model",
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=5,
        help="Number of neighbors for k-NN classifier",
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

    config = MSFRDConfig(
        time_in=args.time_in,
        time_out=args.time_out,
        hidden_dim=args.hidden_dim,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        k_neighbors=args.k_neighbors,
        random_state=args.seed,
    )

    if config.time_in + config.time_out > T:
        raise ValueError(f"time_in + time_out = {config.time_in + config.time_out} exceeds window length T={T}")

    # Set random seeds
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # 1) Train mutation model on healthy training windows
    print("\nTraining MSFRD-style mutation model on healthy training windows...")
    mutation_model = train_mutation_model(X_train, y_train, config, device)

    # 2) Extract mutation features for all splits
    print("\nExtracting mutation features for train / val / test...")
    mut_train, rarity = extract_mutation_features(mutation_model, X_train, config, device)
    mut_val,   _      = extract_mutation_features(mutation_model, X_val,   config, device)
    mut_test,  _      = extract_mutation_features(mutation_model, X_test,  config, device)

    print("\nMutation feature shapes (samples × F*time_out):")
    print(f"  train: {mut_train.shape}")
    print(f"  val  : {mut_val.shape}")
    print(f"  test : {mut_test.shape}")
    print(f"Rarity weights per attribute (F={rarity.shape[0]}):")
    print(rarity)

    # 3) Train similarity-based classifier (k-NN) on training mutations
    print("\nTraining k-NN similarity classifier on mutation features...")
    knn, scale = train_similarity_classifier(mut_train, y_train, rarity, config)

    # 4) Validation evaluation (full, unbalanced)
    val_proba = predict_similarity_classifier(knn, mut_val, scale)
    evaluate_split("Validation (unbalanced)", val_proba, y_val)

    # 5) Balanced test subset and evaluation
    idx_bal = balanced_indices(y_test, random_state=config.random_state)
    print("\nBalanced test subset:")
    print(f"  #failed windows (label=1):  {np.sum(y_test[idx_bal] == 1)}")
    print(f"  #healthy windows (label=0): {np.sum(y_test[idx_bal] == 0)}")
    print(f"  Total windows:              {len(idx_bal)}")

    mut_test_bal = mut_test[idx_bal]
    y_test_bal = y_test[idx_bal]

    te_proba_bal = predict_similarity_classifier(knn, mut_test_bal, scale)
    evaluate_split("Test (balanced: 1:1 healthy:failed)", te_proba_bal, y_test_bal)


if __name__ == "__main__":
    main()
