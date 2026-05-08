import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

DATASET_NAME = "MB2"
ROUND = 1
PROCESSED_ROOT = Path("data/processed")
ARTIFACTS_ROOT = Path("data/artifacts")
PATCH_POLICY = "truncate"

# Where the window files live
WINDOW_ROOT = PROCESSED_ROOT / f"{DATASET_NAME}_round{ROUND}"
TRAIN_PATH = WINDOW_ROOT / "train.npz"
VAL_PATH   = WINDOW_ROOT / "val.npz"
TEST_PATH  = WINDOW_ROOT / "test.npz"

# Where to store learned artifacts
ARTIFACT_ROOT = ARTIFACTS_ROOT / f"{DATASET_NAME}_round{ROUND}"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

SCALER_PATH      = ARTIFACT_ROOT / "mb2_scaler.npz"
ATTR_ENCODER_PT  = ARTIFACT_ROOT / "mb2_attr_autoencoder.pt"
CROSS_ENCODER_PT = ARTIFACT_ROOT / "mb2_cross_autoencoder.pt"
PROTOTYPE_PATH   = ARTIFACT_ROOT / "mb2_prototypes.npz"

# Self-supervised patch config
PATCH_LEN_ATTR  = 5   # days per 1D patch
PATCH_LEN_CROSS = 5   # days per 2D patch
PATCH_STRIDE    = 1   # move 1 day at a time

ATTR_EMB_DIM    = 64
CROSS_EMB_DIM   = 64

N_ATTR_CLUSTERS  = 16
N_CROSS_CLUSTERS = 8

# For KMeans, we don't need *all* patches. Put an upper bound.
# MAX_PATCHES_KMEANS = 1_000_000
# How many random patches to see per epoch
ATTR_SAMPLES_PER_EPOCH  = 2_000_000   # random 1D patches / epoch
CROSS_SAMPLES_PER_EPOCH = 1_000_000   # random 2D patches / epoch

MAX_PATCHES_KMEANS = 1_000_000  # how many embeddings to collect for K-means

BATCH_SIZE_ATTR  = 1024
BATCH_SIZE_CROSS = 512
NUM_EPOCHS_ATTR  = 5
NUM_EPOCHS_CROSS = 5
LR               = 1e-3


def dataset_prefix(dataset_name: str) -> str:
    return dataset_name.strip().lower()


def configure_runtime(
    dataset_name: str,
    round_id: int,
    processed_root: str,
    artifacts_root: str,
    window_root: Optional[str],
    artifact_root: Optional[str],
    patch_len_attr: int,
    patch_len_cross: int,
    patch_policy: str,
) -> None:
    global DATASET_NAME
    global ROUND
    global PROCESSED_ROOT
    global ARTIFACTS_ROOT
    global WINDOW_ROOT
    global TRAIN_PATH
    global VAL_PATH
    global TEST_PATH
    global ARTIFACT_ROOT
    global SCALER_PATH
    global ATTR_ENCODER_PT
    global CROSS_ENCODER_PT
    global PROTOTYPE_PATH
    global PATCH_LEN_ATTR
    global PATCH_LEN_CROSS
    global PATCH_POLICY

    DATASET_NAME = dataset_name.strip().upper()
    ROUND = int(round_id)
    PROCESSED_ROOT = Path(processed_root)
    ARTIFACTS_ROOT = Path(artifacts_root)
    PATCH_LEN_ATTR = int(patch_len_attr)
    PATCH_LEN_CROSS = int(patch_len_cross)
    PATCH_POLICY = patch_policy.strip().lower()

    if PATCH_LEN_ATTR <= 0 or PATCH_LEN_CROSS <= 0:
        raise ValueError("Patch lengths must be positive integers.")
    if PATCH_POLICY != "truncate":
        raise ValueError(
            f"Unsupported patch policy '{PATCH_POLICY}'. Only 'truncate' is implemented."
        )

    if window_root:
        WINDOW_ROOT = Path(window_root)
    else:
        WINDOW_ROOT = PROCESSED_ROOT / f"{DATASET_NAME}_round{ROUND}"

    TRAIN_PATH = WINDOW_ROOT / "train.npz"
    VAL_PATH = WINDOW_ROOT / "val.npz"
    TEST_PATH = WINDOW_ROOT / "test.npz"

    if artifact_root:
        ARTIFACT_ROOT = Path(artifact_root)
    else:
        ARTIFACT_ROOT = ARTIFACTS_ROOT / f"{DATASET_NAME}_round{ROUND}"
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    prefix = dataset_prefix(DATASET_NAME)
    SCALER_PATH = ARTIFACT_ROOT / f"{prefix}_scaler.npz"
    ATTR_ENCODER_PT = ARTIFACT_ROOT / f"{prefix}_attr_autoencoder.pt"
    CROSS_ENCODER_PT = ARTIFACT_ROOT / f"{prefix}_cross_autoencoder.pt"
    PROTOTYPE_PATH = ARTIFACT_ROOT / f"{prefix}_prototypes.npz"


def prototype_assignment_path(split_name: str) -> Path:
    split_norm = split_name.strip().lower()
    if split_norm not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split '{split_name}'. Use train, val, or test.")
    return ARTIFACT_ROOT / f"{dataset_prefix(DATASET_NAME)}_{split_norm}_prototypes.npz"


def count_non_overlapping_patches(total_steps: int, patch_len: int) -> int:
    return total_steps // patch_len


def dropped_trailing_days(total_steps: int, patch_len: int) -> int:
    return total_steps - count_non_overlapping_patches(total_steps, patch_len) * patch_len


# ---------------------------------------------------------
# DATASETS: PATCHES ON THE FLY
# ---------------------------------------------------------

class AttrPatchDataset(Dataset):
    """
    Stochastic 1D attribute patch dataset.

    Instead of enumerating all patches, each item is:
      - sample a random window
      - sample a random attribute
      - sample a random temporal position
    so each epoch sees ATTR_SAMPLES_PER_EPOCH random patches.

    Input X: [N, T, F], patch shape: [1, L_attr].
    """

    def __init__(self, X: np.ndarray, patch_len: int, samples_per_epoch: int, stride: int = 1):
        super().__init__()
        assert X.ndim == 3
        self.X = torch.from_numpy(X.astype(np.float32))  # [N, T, F]
        self.patch_len = patch_len
        self.stride = stride
        self.samples_per_epoch = samples_per_epoch

        self.N, self.T, self.F = self.X.shape
        self.num_pos = (self.T - self.patch_len) // self.stride + 1

    def __len__(self):
        # "virtual" dataset size = how many random patches we want per epoch
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # idx is ignored; we sample randomly
        w = np.random.randint(self.N)        # window index
        f = np.random.randint(self.F)        # feature index
        pos = np.random.randint(self.num_pos)

        start = pos * self.stride
        end = start + self.patch_len

        patch = self.X[w, start:end, f]      # [L_attr]
        patch = patch.unsqueeze(0)           # [1, L_attr]
        return patch



class CrossPatchDataset(Dataset):
    """
    Stochastic 2D cross-attribute patch dataset.

    Each item:
      - sample a random window
      - sample a random temporal position
      - patch = [F, L_cross] (transposed), returned as [1, F, L_cross]
    """

    def __init__(self, X: np.ndarray, patch_len: int, samples_per_epoch: int, stride: int = 1):
        super().__init__()
        assert X.ndim == 3
        self.X = torch.from_numpy(X.astype(np.float32))  # [N, T, F]
        self.patch_len = patch_len
        self.stride = stride
        self.samples_per_epoch = samples_per_epoch

        self.N, self.T, self.F = self.X.shape
        self.num_pos = (self.T - self.patch_len) // self.stride + 1

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        w = np.random.randint(self.N)        # window index
        pos = np.random.randint(self.num_pos)

        start = pos * self.stride
        end = start + self.patch_len

        patch = self.X[w, start:end, :]      # [L_cross, F]
        patch = patch.transpose(0, 1)        # [F, L_cross]
        patch = patch.unsqueeze(0)           # [1, F, L_cross]
        return patch



# ---------------------------------------------------------
# MODELS (AUTOENCODERS)
# ---------------------------------------------------------

class AttrAutoencoder(nn.Module):
    """
    Simple 1D conv autoencoder for attribute patches.

    Input: [B, 1, L_attr]
    Output: reconstruction [B, 1, L_attr] and embedding [B, D]
    """

    def __init__(self, patch_len: int, emb_dim: int = 64):
        super().__init__()
        self.patch_len = patch_len
        self.emb_dim = emb_dim

        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.encoder_fc = nn.Linear(32 * patch_len, emb_dim)

        self.decoder_fc = nn.Linear(emb_dim, 32 * patch_len)
        self.decoder_cnn = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, L]
        h = self.encoder_cnn(x)               # [B, 32, L]
        h = h.view(h.size(0), -1)             # [B, 32*L]
        z = self.encoder_fc(h)                # [B, D]
        return z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        h = self.decoder_fc(z)                # [B, 32*L]
        h = h.view(x.size(0), 32, self.patch_len)
        recon = self.decoder_cnn(h)           # [B, 1, L]
        return recon, z


class CrossAutoencoder(nn.Module):
    """
    2D conv autoencoder for cross-attribute patches.

    Input: [B, 1, F, L_cross]
    Output: reconstruction [B, 1, F, L_cross] and embedding [B, D]
    """

    def __init__(self, num_features: int, patch_len: int, emb_dim: int = 64):
        super().__init__()
        self.num_features = num_features
        self.patch_len = patch_len
        self.emb_dim = emb_dim

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.encoder_fc = nn.Linear(32 * num_features * patch_len, emb_dim)

        self.decoder_fc = nn.Linear(emb_dim, 32 * num_features * patch_len)
        self.decoder_cnn = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, F, L]
        h = self.encoder_cnn(x)                         # [B, 32, F, L]
        h = h.view(h.size(0), -1)                       # [B, 32*F*L]
        z = self.encoder_fc(h)                          # [B, D]
        return z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        h = self.decoder_fc(z)                          # [B, 32*F*L]
        h = h.view(x.size(0), 32, self.num_features, self.patch_len)
        recon = self.decoder_cnn(h)                     # [B, 1, F, L]
        return recon, z


# ---------------------------------------------------------
# TRAINING UTIL
# ---------------------------------------------------------
# remove the old global DEVICE line

def get_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)

def train_autoencoder(model, train_loader, val_loader, num_epochs, lr, device, name):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        if val_loader is not None:
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    recon, _ = model(batch)
                    loss = criterion(recon, batch)
                    val_total += loss.item() * batch.size(0)
            val_loss = val_total / len(val_loader.dataset)
            print(f"[{name}] Epoch {epoch}/{num_epochs} | "
                  f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        else:
            print(f"[{name}] Epoch {epoch}/{num_epochs} | "
                  f"train_loss={train_loss:.4f}")


# ---------------------------------------------------------
# STEP 1: OFFLINE PROTOTYPE LEARNING
# ---------------------------------------------------------

def run_step1(device):
    print(f"=== STEP 1: Offline self-supervised prototype learning ({DATASET_NAME}) ===")
    print(
        f"Window root: {WINDOW_ROOT} | Artifact root: {ARTIFACT_ROOT} | "
        f"L_attr={PATCH_LEN_ATTR} | L_cross={PATCH_LEN_CROSS} | patch_policy={PATCH_POLICY}"
    )

    # ---------- Load train/val ----------
    train_npz = np.load(TRAIN_PATH)
    X_train = train_npz["X"]  # [N, 30, F]
    feature_names = train_npz["features"].tolist()
    N_train, T, F = X_train.shape
    print(f"Train windows: {N_train}, T={T}, F={F}")

    val_npz = np.load(VAL_PATH)
    X_val = val_npz["X"]
    print(f"Val windows: {X_val.shape[0]}")

    # ---------- Feature normalization ----------
    feat_mean = X_train.mean(axis=(0, 1), keepdims=True)   # [1,1,F]
    feat_std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    X_train_norm = (X_train - feat_mean) / feat_std
    X_val_norm   = (X_val   - feat_mean) / feat_std

    np.savez(
        SCALER_PATH,
        mean=feat_mean,
        std=feat_std,
        features=np.array(feature_names),
    )
    print(f"Saved scaler to {SCALER_PATH}")

    # ---------- Patch datasets ----------
    attr_train_ds = AttrPatchDataset(
        X_train_norm,
        patch_len=PATCH_LEN_ATTR,
        samples_per_epoch=ATTR_SAMPLES_PER_EPOCH,
        stride=PATCH_STRIDE,
    )
    attr_val_ds = AttrPatchDataset(
        X_val_norm,
        patch_len=PATCH_LEN_ATTR,
        samples_per_epoch=ATTR_SAMPLES_PER_EPOCH // 4,
        stride=PATCH_STRIDE,
    )

    cross_train_ds = CrossPatchDataset(
        X_train_norm,
        patch_len=PATCH_LEN_CROSS,
        samples_per_epoch=CROSS_SAMPLES_PER_EPOCH,
        stride=PATCH_STRIDE,
    )
    cross_val_ds = CrossPatchDataset(
        X_val_norm,
        patch_len=PATCH_LEN_CROSS,
        samples_per_epoch=CROSS_SAMPLES_PER_EPOCH // 4,
        stride=PATCH_STRIDE,
    )

    print(f"Attr patches per epoch:  {len(attr_train_ds)} train, {len(attr_val_ds)} val")
    print(f"Cross patches per epoch: {len(cross_train_ds)} train, {len(cross_val_ds)} val")

    # ---------- Train autoencoders ----------
    attr_ae = AttrAutoencoder(patch_len=PATCH_LEN_ATTR, emb_dim=ATTR_EMB_DIM)
    cross_ae = CrossAutoencoder(num_features=F, patch_len=PATCH_LEN_CROSS, emb_dim=CROSS_EMB_DIM)

    attr_train_loader = DataLoader(
        attr_train_ds, batch_size=BATCH_SIZE_ATTR,
        shuffle=True, num_workers=4, pin_memory=True
    )
    attr_val_loader = DataLoader(
        attr_val_ds, batch_size=BATCH_SIZE_ATTR,
        shuffle=False, num_workers=4, pin_memory=True
    )
    cross_train_loader = DataLoader(
        cross_train_ds, batch_size=BATCH_SIZE_CROSS,
        shuffle=True, num_workers=4, pin_memory=True
    )
    cross_val_loader = DataLoader(
        cross_val_ds, batch_size=BATCH_SIZE_CROSS,
        shuffle=False, num_workers=4, pin_memory=True
    )

    train_autoencoder(
        attr_ae, attr_train_loader, attr_val_loader,
        NUM_EPOCHS_ATTR, LR, device, name="AttrAE"
    )
    train_autoencoder(
        cross_ae, cross_train_loader, cross_val_loader,
        NUM_EPOCHS_CROSS, LR, device, name="CrossAE"
    )

    torch.save(attr_ae.state_dict(), ATTR_ENCODER_PT)
    torch.save(cross_ae.state_dict(), CROSS_ENCODER_PT)
    print(f"Saved attribute autoencoder to {ATTR_ENCODER_PT}")
    print(f"Saved cross autoencoder to {CROSS_ENCODER_PT}")

    # ---------- Extract embeddings for K-means ----------
    def collect_embeddings(model, dataset, batch_size, name, device):
        model.to(device)
        model.eval()
        loader = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=True, num_workers=4, pin_memory=True
        )
        embs = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Collecting embeddings ({name})"):
                batch = batch.to(device)
                z = model.encode(batch)          # [B, D]
                embs.append(z.cpu().numpy())
                if sum(x.shape[0] for x in embs) >= MAX_PATCHES_KMEANS:
                    break
        embs = np.concatenate(embs, axis=0)
        print(f"{name}: collected {embs.shape[0]} embeddings")
        return embs

    attr_embs = collect_embeddings(attr_ae, attr_train_ds, BATCH_SIZE_ATTR,  "Attr",  device)
    cross_embs = collect_embeddings(cross_ae, cross_train_ds, BATCH_SIZE_CROSS, "Cross", device)

    # ---------- K-means for prototypes ----------
    print("Running KMeans for attribute prototypes...")
    kmeans_attr = KMeans(n_clusters=N_ATTR_CLUSTERS, random_state=0, n_init="auto")
    kmeans_attr.fit(attr_embs)
    attr_centers = kmeans_attr.cluster_centers_  # [K_attr, D_attr]

    print("Running KMeans for cross prototypes...")
    kmeans_cross = KMeans(n_clusters=N_CROSS_CLUSTERS, random_state=0, n_init="auto")
    kmeans_cross.fit(cross_embs)
    cross_centers = kmeans_cross.cluster_centers_  # [K_cross, D_cross]

    # ---------- Helper: collect labeled patch distances ----------
    def collect_labeled_patch_dists(X_norm, y_labels):
        """
        Compute per-patch min distance to nearest center + patch labels.

        X_norm:   [N, T, F] normalized windows
        y_labels: [N] binary labels (0=healthy, 1=failure)
        Returns:
            attr_dists_all, attr_labels_all, cross_dists_all, cross_labels_all
        """
        X_norm = np.asarray(X_norm, dtype=np.float32)
        y_labels = np.asarray(y_labels).astype(int).reshape(-1)
        N_local, T_local, F_local = X_norm.shape
        assert y_labels.shape[0] == N_local, "Label length must match number of windows."

        attr_dists_list, attr_labels_list = [], []
        cross_dists_list, cross_labels_list = [], []

        for n in tqdm(range(N_local), desc="Collecting patch distances for threshold tuning"):
            label_n = int(y_labels[n])
            win = X_norm[n]  # [T, F]

            # 1D attribute patches
            for f_idx in range(F_local):
                series = win[:, f_idx]  # [T]
                series_t = torch.as_tensor(series, dtype=torch.float32, device=device)
                patches = series_t.unfold(0, PATCH_LEN_ATTR, PATCH_LEN_ATTR)  # [P, L]
                if patches.numel() == 0:
                    continue
                patches = patches.unsqueeze(1)  # [P, 1, L]
                with torch.no_grad():
                    emb = attr_ae.encode(patches).cpu().numpy()  # [P, D]
                dists = np.linalg.norm(
                    emb[:, None, :] - attr_centers[None, :, :],
                    axis=-1,
                )  # [P, K_attr]
                min_d = dists.min(axis=1)  # [P]
                attr_dists_list.append(min_d)
                attr_labels_list.append(np.full_like(min_d, label_n, dtype=np.int8))

            # 2D cross-attribute patches
            series = win  # [T, F]
            series_t = torch.as_tensor(series, dtype=torch.float32, device=device)
            patches = series_t.unfold(0, PATCH_LEN_CROSS, PATCH_LEN_CROSS)  # [P, L, F]
            if patches.numel() > 0:
                patches = patches.permute(0, 2, 1)  # [P, F, L]
                patches = patches.unsqueeze(1)      # [P, 1, F, L]
                with torch.no_grad():
                    emb = cross_ae.encode(patches).cpu().numpy()  # [P, D]
                dists = np.linalg.norm(
                    emb[:, None, :] - cross_centers[None, :, :],
                    axis=-1,
                )  # [P, K_cross]
                min_d = dists.min(axis=1)
                cross_dists_list.append(min_d)
                cross_labels_list.append(np.full_like(min_d, label_n, dtype=np.int8))

        attr_dists_all  = np.concatenate(attr_dists_list,  axis=0) if attr_dists_list  else np.empty(0)
        attr_labels_all = np.concatenate(attr_labels_list, axis=0) if attr_labels_list else np.empty(0, dtype=np.int8)
        cross_dists_all  = np.concatenate(cross_dists_list,  axis=0) if cross_dists_list  else np.empty(0)
        cross_labels_all = np.concatenate(cross_labels_list, axis=0) if cross_labels_list else np.empty(0, dtype=np.int8)
        return attr_dists_all, attr_labels_all, cross_dists_all, cross_labels_all

    # ---------- Helper: heuristic-3 threshold tuning ----------
    def tune_threshold_heuristic3(dists, labels, max_fpr=0.05, name="attr"):
        """
        Heuristic 3:
          - sweep quantiles
          - for each tau: compute FPR = P(novel | healthy), TPR = P(novel | failure)
          - keep tau with FPR <= max_fpr and best F1.
        """
        dists = np.asarray(dists, dtype=np.float32).reshape(-1)
        labels = np.asarray(labels).astype(int).reshape(-1)
        if dists.size == 0:
            raise ValueError(f"[{name}] No distances provided for threshold tuning.")

        healthy_mask = labels == 0
        fail_mask    = labels == 1

        if not healthy_mask.any() or not fail_mask.any():
            print(f"[{name}] WARNING: labels missing one of the classes; falling back to 95th percentile.")
            return float(np.quantile(dists, 0.95))

        quantiles = np.linspace(0.90, 0.995, 12)
        best_tau = None
        best_f1  = -1.0
        best_stats = None

        for q in quantiles:
            tau = float(np.quantile(dists, q))
            novel = dists > tau

            fpr = novel[healthy_mask].mean() if healthy_mask.any() else 0.0
            tpr = novel[fail_mask].mean()    if fail_mask.any()    else 0.0

            # constraint on false alarms
            if fpr > max_fpr:
                continue

            tp = np.logical_and(novel,     fail_mask).sum()
            fp = np.logical_and(novel,     healthy_mask).sum()
            fn = np.logical_and(~novel,    fail_mask).sum()

            precision = tp / (tp + fp + 1e-9)
            recall    = tp / (tp + fn + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)

            if f1 > best_f1:
                best_f1 = f1
                best_tau = tau
                best_stats = dict(q=q, fpr=float(fpr), tpr=float(tpr),
                                  precision=float(precision), recall=float(recall))

        if best_tau is None:
            # No quantile satisfied FPR constraint; fall back to the highest quantile
            print(f"[{name}] WARNING: no quantile satisfied FPR <= {max_fpr:.3f}; "
                  f"falling back to highest quantile.")
            best_tau = float(np.quantile(dists, quantiles[-1]))
            best_stats = dict(q=quantiles[-1], fpr=float("nan"), tpr=float("nan"),
                              precision=float("nan"), recall=float("nan"))

        print(
            f"[{name}] chosen tau={best_tau:.4f} "
            f"(q={best_stats['q']:.3f}, F1={best_f1:.4f}, "
            f"FPR={best_stats['fpr']:.4f}, TPR={best_stats['tpr']:.4f})"
        )
        return best_tau

    # ---------- Calibrate novelty thresholds ----------
    attr_threshold = None
    cross_threshold = None

    # Try to find labels in train_npz
    label_keys = [k for k in ["y", "labels", "y_fail", "label"] if k in train_npz.files]
    if label_keys:
        label_key = label_keys[0]
        y_train = train_npz[label_key]
        print(f"Found train labels in key '{label_key}' with shape {y_train.shape}.")

        (attr_dists_all,
         attr_labels_all,
         cross_dists_all,
         cross_labels_all) = collect_labeled_patch_dists(X_train_norm, y_train)

        attr_threshold  = tune_threshold_heuristic3(attr_dists_all,  attr_labels_all,
                                                    max_fpr=0.05, name="attr")
        cross_threshold = tune_threshold_heuristic3(cross_dists_all, cross_labels_all,
                                                    max_fpr=0.05, name="cross")
    else:
        print("WARNING: no labels found in TRAIN_PATH; using simple 95th percentile thresholds.")
        attr_dists  = kmeans_attr.transform(attr_embs).min(axis=1)
        cross_dists = kmeans_cross.transform(cross_embs).min(axis=1)
        attr_threshold  = float(np.quantile(attr_dists,  0.95))
        cross_threshold = float(np.quantile(cross_dists, 0.95))

    # ---------- Save prototype memory ----------
    np.savez_compressed(
        PROTOTYPE_PATH,
        attr_centers=attr_centers,
        cross_centers=cross_centers,
        attr_threshold=np.array([attr_threshold], dtype=np.float32),
        cross_threshold=np.array([cross_threshold], dtype=np.float32),
        patch_policy=np.array([PATCH_POLICY]),
        patch_len_attr=np.array([PATCH_LEN_ATTR]),
        patch_len_cross=np.array([PATCH_LEN_CROSS]),
        attr_emb_dim=np.array([ATTR_EMB_DIM]),
        cross_emb_dim=np.array([CROSS_EMB_DIM]),
        n_attr_clusters=np.array([N_ATTR_CLUSTERS]),
        n_cross_clusters=np.array([N_CROSS_CLUSTERS]),
        feature_names=np.array(feature_names),
    )
    print(f"Saved prototype memory to {PROTOTYPE_PATH}")
    print("=== STEP 1 done ===")


# ---------------------------------------------------------
# STEP 2: INFERENCE ON TEST WINDOWS
# ---------------------------------------------------------

def run_step2(device, split_name: str = "test"):
    split_norm = split_name.strip().lower()
    print(f"=== STEP 2: Prototype assignment on {split_norm} windows ({DATASET_NAME}) ===")

    # ---------- Load artifacts ----------
    scaler = np.load(SCALER_PATH)
    feat_mean = scaler["mean"]
    feat_std  = scaler["std"]
    feature_names = scaler["features"].tolist()
    F = len(feature_names)

    proto = np.load(PROTOTYPE_PATH)
    attr_centers   = proto["attr_centers"]
    cross_centers  = proto["cross_centers"]
    attr_threshold  = float(proto["attr_threshold"][0])
    cross_threshold = float(proto["cross_threshold"][0])
    patch_len_attr  = int(proto["patch_len_attr"][0])
    patch_len_cross = int(proto["patch_len_cross"][0])
    patch_policy = str(proto["patch_policy"][0]) if "patch_policy" in proto.files else "truncate"
    if patch_policy != "truncate":
        raise ValueError(
            f"Unsupported patch policy '{patch_policy}' stored in {PROTOTYPE_PATH}."
        )

    # ---------- Load encoders ----------
    attr_ae = AttrAutoencoder(patch_len=patch_len_attr, emb_dim=ATTR_EMB_DIM)
    attr_ae.load_state_dict(torch.load(ATTR_ENCODER_PT, map_location=device))
    attr_ae.to(device).eval()

    cross_ae = CrossAutoencoder(num_features=F, patch_len=patch_len_cross, emb_dim=CROSS_EMB_DIM)
    cross_ae.load_state_dict(torch.load(CROSS_ENCODER_PT, map_location=device))
    cross_ae.to(device).eval()

    # ---------- Load requested split windows ----------
    split_path_map = {
        "train": TRAIN_PATH,
        "val": VAL_PATH,
        "test": TEST_PATH,
    }
    split_path = split_path_map[split_norm]
    split_npz = np.load(split_path)
    X_split = split_npz["X"]
    N, T, F_test = X_split.shape
    assert F_test == F, "Feature mismatch between scaler and requested split."
    print(f"{split_norm.capitalize()} windows: {N}, T={T}, F={F}")

    # standardize
    X_split_norm = (X_split - feat_mean) / feat_std

    # --- use explicit non-overlapping truncation, matching the training/inference path ---
    num_pos_attr = count_non_overlapping_patches(T, patch_len_attr)
    num_pos_cross = count_non_overlapping_patches(T, patch_len_cross)
    dropped_attr_days = dropped_trailing_days(T, patch_len_attr)
    dropped_cross_days = dropped_trailing_days(T, patch_len_cross)
    if dropped_attr_days or dropped_cross_days:
        print(
            "Using explicit truncation for non-divisible patch lengths: "
            f"drop {dropped_attr_days} trailing day(s) for attribute patches and "
            f"{dropped_cross_days} trailing day(s) for cross-attribute patches."
        )

    # Output arrays:
    attr_protos  = -np.ones((N, F, num_pos_attr),    dtype=np.int32)
    attr_novel   =  np.zeros_like(attr_protos,       dtype=bool)
    cross_protos = -np.ones((N,     num_pos_cross),  dtype=np.int32)
    cross_novel  =  np.zeros_like(cross_protos,      dtype=bool)

    # ---------- Attribute prototypes ----------
    print("Assigning attribute prototypes...")
    for n in tqdm(range(N), desc="Attr prototypes per window"):
        for f_idx in range(F):
            series = X_split_norm[n, :, f_idx]  # [T]
            series_t = torch.as_tensor(series, dtype=torch.float32, device=device)

            patches = series_t.unfold(0, patch_len_attr, patch_len_attr)  # [P, L]
            if patches.numel() == 0:
                continue
            patches = patches.unsqueeze(1)  # [P, 1, L]

            with torch.no_grad():
                emb = attr_ae.encode(patches).cpu().numpy()  # [P, D]

            dists = np.linalg.norm(
                emb[:, None, :] - attr_centers[None, :, :],
                axis=-1,
            )  # [P, K_attr]
            proto_ids = dists.argmin(axis=1)
            min_dists = dists.min(axis=1)

            attr_protos[n, f_idx, :] = proto_ids
            attr_novel[n, f_idx, :]  = min_dists > attr_threshold

    # ---------- Cross prototypes ----------
    print("Assigning cross prototypes...")
    for n in tqdm(range(N), desc="Cross prototypes per window"):
        series = X_split_norm[n]  # [T, F]
        series_t = torch.as_tensor(series, dtype=torch.float32, device=device)

        patches = series_t.unfold(0, patch_len_cross, patch_len_cross)  # [P, L, F]
        if patches.numel() == 0:
            continue
        patches = patches.permute(0, 2, 1).unsqueeze(1)  # [P, 1, F, L]

        with torch.no_grad():
            emb = cross_ae.encode(patches).cpu().numpy()  # [P, D]

        dists = np.linalg.norm(
            emb[:, None, :] - cross_centers[None, :, :],
            axis=-1,
        )  # [P, K_cross]
        proto_ids = dists.argmin(axis=1)
        min_dists = dists.min(axis=1)

        cross_protos[n, :] = proto_ids
        cross_novel[n, :]  = min_dists > cross_threshold

    # ---------- Save outputs ----------
    split_out_path = prototype_assignment_path(split_norm)
    np.savez_compressed(
        split_out_path,
        attr_protos=attr_protos,
        attr_novel=attr_novel,
        cross_protos=cross_protos,
        cross_novel=cross_novel,
        patch_policy=np.array([patch_policy]),
        patch_len_attr=np.array([patch_len_attr]),
        patch_len_cross=np.array([patch_len_cross]),
        window_days=np.array([T]),
        covered_days_attr=np.array([num_pos_attr * patch_len_attr]),
        covered_days_cross=np.array([num_pos_cross * patch_len_cross]),
        dropped_days_attr=np.array([dropped_attr_days]),
        dropped_days_cross=np.array([dropped_cross_days]),
        feature_names=np.array(feature_names),
    )
    print(f"Saved {split_norm} prototype assignments to {split_out_path}")
    print("=== STEP 2 done ===")

# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["step1", "step2"],
        required=True,
        help="step1 = offline prototype learning, step2 = inference on test set",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device to use, e.g., 'cuda:0' (A100) or 'cpu'",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DATASET_NAME,
        help="Dataset/model folder name, e.g. MB1 or MB2.",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=ROUND,
        help="Temporal split round id.",
    )
    parser.add_argument(
        "--processed-root",
        type=str,
        default=str(PROCESSED_ROOT),
        help="Root directory containing processed split folders.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=str,
        default=str(ARTIFACTS_ROOT),
        help="Root directory where encoder/prototype artifacts are stored.",
    )
    parser.add_argument(
        "--window-root",
        type=str,
        default=None,
        help="Optional explicit directory containing train.npz/val.npz/test.npz.",
    )
    parser.add_argument(
        "--artifact-root",
        type=str,
        default=None,
        help="Optional explicit artifact directory for this run.",
    )
    parser.add_argument(
        "--patch-len-attr",
        type=int,
        default=PATCH_LEN_ATTR,
        help="Attribute-level patch length in days.",
    )
    parser.add_argument(
        "--patch-len-cross",
        type=int,
        default=PATCH_LEN_CROSS,
        help="Cross-attribute patch length in days.",
    )
    parser.add_argument(
        "--patch-policy",
        type=str,
        default=PATCH_POLICY,
        help="Non-overlapping patch policy. Only 'truncate' is implemented.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split to assign when mode=step2.",
    )
    args = parser.parse_args()

    configure_runtime(
        dataset_name=args.dataset_name,
        round_id=args.round,
        processed_root=args.processed_root,
        artifacts_root=args.artifacts_root,
        window_root=args.window_root,
        artifact_root=args.artifact_root,
        patch_len_attr=args.patch_len_attr,
        patch_len_cross=args.patch_len_cross,
        patch_policy=args.patch_policy,
    )

    device = get_device(args.device)
    print(f"Using device: {device}")

    if args.mode == "step1":
        run_step1(device)
    elif args.mode == "step2":
        run_step2(device, split_name=args.split)



if __name__ == "__main__":
    main()
