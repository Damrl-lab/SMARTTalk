#!/usr/bin/env python3
"""
LSTM baseline for SSD failure prediction (binary classification).

Assumptions about data:
- data_dir contains train.npz, val.npz, test.npz
- Each .npz has at least:
    X: shape (N, T, F) or (N, F, T) with T=30 days
    y: shape (N,) with labels 0 = healthy, 1 = failed

We:
- Balance healthy vs failed by undersampling healthy to match #failed
- Train a simple LSTM classifier on 30-day sequences
- Report precision, recall, F1 on a balanced test set
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


# -------------------------
# Utility / data helpers
# -------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split_npz(path):
    data = np.load(path)
    if "X" not in data or "y" not in data:
        raise ValueError(f"{path} must contain 'X' and 'y' arrays")
    X = data["X"]
    y = data["y"]
    return X, y


def infer_sequence_layout(X, expected_seq_len=30):
    """
    Ensure X has shape (N, T, F) where T is time (e.g., 30 days).
    If X is (N, F, T), transpose to (N, T, F).
    If none of dims equals expected_seq_len, assume current layout is already (N, T, F).
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X.ndim == 3, got {X.ndim}")

    N, d1, d2 = X.shape
    # Try to detect time dimension as 'expected_seq_len'
    if d1 == expected_seq_len and d2 != expected_seq_len:
        # Already (N, T, F)
        return X
    elif d2 == expected_seq_len and d1 != expected_seq_len:
        # (N, F, T) -> (N, T, F)
        return np.transpose(X, (0, 2, 1))
    else:
        # Fallback: assume d1 is time, d2 is feature
        return X


def make_binary_and_balanced(X, y, seed=0, healthy_per_fail=1.0):
    """
    Filter to labels {0,1} and sample healthy (0) relative to failed (1).
    Returns ratio-based X, y (shuffled).
    """
    # Filter to valid binary labels
    valid_mask = np.isin(y, [0, 1])
    X = X[valid_mask]
    y = y[valid_mask]

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos == 0:
        raise ValueError("No failed samples (y==1) found in this split.")
    if n_neg == 0:
        raise ValueError("No healthy samples (y==0) found in this split.")

    # Sample healthy windows relative to failed windows
    rng = np.random.default_rng(seed)
    target_neg = max(1, int(round(n_pos * healthy_per_fail)))
    if n_neg >= target_neg:
        sampled_neg_idx = rng.choice(neg_idx, size=target_neg, replace=False)
        sampled_pos_idx = pos_idx
    else:
        sampled_neg_idx = neg_idx
        max_pos = max(1, int(np.floor(n_neg / healthy_per_fail)))
        sampled_pos_idx = pos_idx if max_pos >= n_pos else rng.choice(pos_idx, size=max_pos, replace=False)

    all_idx = np.concatenate([sampled_pos_idx, sampled_neg_idx])
    rng.shuffle(all_idx)

    X_bal = X[all_idx]
    y_bal = y[all_idx]

    return X_bal, y_bal


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------------
# Model
# -------------------------

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)       # out: (B, T, H)
        last = out[:, -1, :]        # (B, H) -> last time step
        logits = self.fc(last)      # (B, 1)
        return logits.squeeze(-1)   # (B,)


# -------------------------
# Train / eval loops
# -------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).float()

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_logits = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(y_batch.numpy())

    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_logits)
    y_pred = (1 / (1 + np.exp(-y_score)) >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1
    )
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
    }
    return metrics


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="LSTM baseline for SSD failure prediction (binary).")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with train.npz, val.npz, test.npz")
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Hidden size of LSTM")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout in LSTM (if num_layers>1)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--seq_len", type=int, default=30,
                        help="Expected sequence length (days per window)")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load data ----
    train_path = os.path.join(args.data_dir, "train.npz")
    val_path   = os.path.join(args.data_dir, "val.npz")
    test_path  = os.path.join(args.data_dir, "test.npz")

    X_train, y_train = load_split_npz(train_path)
    X_val, y_val     = load_split_npz(val_path)
    X_test, y_test   = load_split_npz(test_path)

    # Ensure consistent (N, T, F) layout based on train
    X_train = infer_sequence_layout(X_train, expected_seq_len=args.seq_len)
    X_val   = infer_sequence_layout(X_val, expected_seq_len=args.seq_len)
    X_test  = infer_sequence_layout(X_test, expected_seq_len=args.seq_len)

    N_train, T, F = X_train.shape
    print(f"Train shape (N, T, F): {X_train.shape}")
    print(f"Val   shape (N, T, F): {X_val.shape}")
    print(f"Test  shape (N, T, F): {X_test.shape}")

    # ---- Balance healthy/failed for each split ----
    X_train_bal, y_train_bal = make_binary_and_balanced(X_train, y_train, seed=args.seed)
    X_val_bal, y_val_bal     = make_binary_and_balanced(X_val, y_val, seed=args.seed + 1)
    X_test_bal, y_test_bal   = make_binary_and_balanced(X_test, y_test, seed=args.seed + 2)

    print("\n=== Balanced split sizes ===")
    print(f"Train: {X_train_bal.shape[0]} (failed={np.sum(y_train_bal==1)}, healthy={np.sum(y_train_bal==0)})")
    print(f"Val  : {X_val_bal.shape[0]} (failed={np.sum(y_val_bal==1)}, healthy={np.sum(y_val_bal==0)})")
    print(f"Test : {X_test_bal.shape[0]} (failed={np.sum(y_test_bal==1)}, healthy={np.sum(y_test_bal==0)})")

    # ---- Build datasets/loaders ----
    train_ds = SequenceDataset(X_train_bal, y_train_bal)
    val_ds   = SequenceDataset(X_val_bal, y_val_bal)
    test_ds  = SequenceDataset(X_test_bal, y_test_bal)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # ---- Model, optimizer ----
    model = LSTMClassifier(
        input_size=F,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- Train with simple val-based model selection (by F1) ----
    best_f1 = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        val_f1 = val_metrics["f1"]

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
              f"val_prec={val_metrics['precision']:.4f} "
              f"val_rec={val_metrics['recall']:.4f} "
              f"val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nLoaded best model by val F1 = {best_f1:.4f}")

    # ---- Final evaluation on ratio-based test subset ----
    test_metrics = evaluate(model, test_loader, device)
    cm = test_metrics["confusion_matrix"]

    print("\n=== LSTM baseline: Test results on ratio-based subset ===")
    print(f"Requested samples (ratio-based): {len(test_metrics['y_true'])}")
    print("Confusion matrix (rows: y=1,0; cols: pred=1,0):")
    print(cm)
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall   : {test_metrics['recall']:.4f}")
    print(f"F1-score : {test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
