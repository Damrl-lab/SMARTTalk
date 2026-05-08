import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

# progress bar (optional)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# =========================================================
# Limits for patch collection (per cluster)
# =========================================================

# How many raw patches we keep per prototype for phrase learning.
# These are per-cluster caps; reservoir sampling keeps them representative.
MAX_ATTR_PATCHES_PER_CLUSTER = 50000
MAX_CROSS_PATCHES_PER_CLUSTER = 20000


# =========================================================
# Import encoders from step1.py
# =========================================================

from step1 import AttrAutoencoder, CrossAutoencoder


# =========================================================
# Utility
# =========================================================

def get_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def compute_series_stats(series: np.ndarray) -> Dict[str, float]:
    """
    Compute simple temporal statistics for a 1D patch.
    Used on the aggregated (mean) series of a cluster.
    """
    s = np.asarray(series, dtype=float)
    L = s.size
    if L == 0:
        return {
            "L": 0,
            "level_mean": 0.0,
            "level_std": 0.0,
            "delta": 0.0,
            "amp": 0.0,
            "max_step": 0.0,
            "mean_abs_step": 0.0,
            "step_std": 0.0,
            "pos_max_step": 0.0,
        }

    level_mean = float(s.mean())
    level_std = float(s.std())
    start = float(s[0])
    end = float(s[-1])
    delta = end - start

    centered = s - start
    amp = float(np.max(centered) - np.min(centered))

    if L > 1:
        diffs = np.diff(centered)
        max_step = float(np.max(np.abs(diffs)))
        mean_abs_step = float(np.mean(np.abs(diffs)))
        step_std = float(np.std(diffs))
        pos_max_step = float(np.argmax(np.abs(diffs)) / (L - 1))
    else:
        max_step = 0.0
        mean_abs_step = 0.0
        step_std = 0.0
        pos_max_step = 0.0

    return {
        "L": int(L),
        "level_mean": level_mean,
        "level_std": level_std,
        "delta": float(delta),
        "amp": float(amp),
        "max_step": max_step,
        "mean_abs_step": mean_abs_step,
        "step_std": step_std,
        "pos_max_step": pos_max_step,
    }


# =========================================================
# Attribute-level labels / phrases and heuristics
# =========================================================

ATTR_LABEL_TO_PHRASE = {
    # housekeeping
    "UNUSED_PROTO": "prototype rarely activated in training windows",

    # flat / stable
    "ZERO_OR_NEAR_ZERO": "always zero or near zero throughout the patch",
    "LOW_STABLE": "roughly constant over the patch",
    "HIGH_STABLE": "stays at a high but stable level",

    # smooth trends
    "SLOW_RISE": "slowly increasing over the patch",
    "SLOW_DECAY": "slowly decreasing over the patch",
    "FAST_RISE": "rapidly increasing over the patch",
    "FAST_DECAY": "rapidly decreasing over the patch",
    "SATURATING_RISE": "rises and then levels off near saturation",
    "SATURATING_DECAY": "drops and then levels off at a lower plateau",

    # step-like changes
    "STEP_UP": "sudden jump to a higher stable level",
    "STEP_DOWN": "sudden drop to a lower stable level",

    # spikes and bursts
    "SINGLE_SPIKE_EARLY": "a single sharp spike early in the patch",
    "SINGLE_SPIKE_MID": "a single sharp spike in the middle of the patch",
    "SINGLE_SPIKE_LATE": "a single sharp spike late in the patch",
    "REPEATED_BURSTS": "multiple bursts within the patch",
    "LOW_RATE_BURSTS": "occasional small bursts on top of a stable baseline",
    "HIGH_RATE_BURSTS": "frequent bursts across most of the patch",

    # noisy / irregular
    "NOISY_FLUCTUATIONS": "noisy up-and-down fluctuations",
    "DRIFT_WITH_NOISE": "slow drift with moderate noise around the trend",

    # shape patterns
    "U_SHAPED": "drops and then rises, forming a U-shaped pattern",
    "HUMP_SHAPED": "rises and then falls, forming a hill-shaped pattern",
}


def classify_attr_trend(mean_series: np.ndarray) -> Tuple[str, Dict[str, float]]:
    """
    Map a prototype's aggregated series into one of a small set of trend labels.

    The thresholds are conservative so that a truly flat patch will
    never be mis-labeled as a spike.
    """
    stats = compute_series_stats(mean_series)
    L = stats["L"]
    if L == 0:
        return "UNUSED_PROTO", stats

    s = np.asarray(mean_series, dtype=float)
    if L == 1:
        # Single point: treat as stable.
        if abs(s[0]) < 1e-6:
            return "ZERO_OR_NEAR_ZERO", stats
        return "LOW_STABLE", stats

    # Absolute near-zero check (for error counters that stay at 0).
    max_abs = float(np.max(np.abs(s)))
    if max_abs < 1e-6:
        return "ZERO_OR_NEAR_ZERO", stats

    # Shape-based analysis in a relative-normalised space.
    s0 = s - s[0]
    amp = np.max(np.abs(s0))
    if amp < 1e-12:
        # All points equal to the first value.
        if max_abs < 1.0:
            return "ZERO_OR_NEAR_ZERO", stats
        else:
            return "LOW_STABLE", stats

    norm = s0 / (amp + 1e-12)
    diffs = np.diff(norm)
    noise = float(np.mean(np.abs(diffs)))
    global_trend = float(norm[-1] - norm[0])

    # Positions where the normalised value is close to an extremum.
    SPIKE_LEVEL = 0.7
    spike_idx = np.where(np.abs(norm) >= SPIKE_LEVEL)[0]
    num_spike_pts = int(spike_idx.size)
    spike_frac = spike_idx.mean() / (L - 1) if num_spike_pts > 0 else 0.0

    # Basic thresholds (tuneable).
    STABLE_AMP_EPS = 0.05      # relative amplitude below which we call it stable
    STABLE_NOISE_EPS = 0.06    # low step noise
    TREND_MIN = 0.3            # minimum end–start change for a trend
    NOISY_NOISE_EPS = 0.25

    # ---- stable-ish behaviour ----
    if amp < STABLE_AMP_EPS and noise < STABLE_NOISE_EPS:
        if max_abs < 1.0:
            return "ZERO_OR_NEAR_ZERO", stats
        else:
            return "LOW_STABLE", stats

    # ---- monotone-like behaviour ----
    tol = 0.05
    mono_inc = np.all(diffs >= -tol)
    mono_dec = np.all(diffs <= tol)

    if mono_inc and global_trend > TREND_MIN and noise < NOISY_NOISE_EPS:
        # distinguish slow/fast rise via final delta magnitude
        if global_trend > 0.7:
            return "FAST_RISE", stats
        else:
            return "SLOW_RISE", stats
    if mono_dec and global_trend < -TREND_MIN and noise < NOISY_NOISE_EPS:
        if global_trend < -0.7:
            return "FAST_DECAY", stats
        else:
            return "SLOW_DECAY", stats

    # ---- saturating patterns (rise then plateau, or fall then plateau) ----
    mid = L // 2
    first_half = norm[:mid]
    second_half = norm[mid:]
    if len(first_half) > 0 and len(second_half) > 0:
        fh_trend = float(first_half[-1] - first_half[0])
        sh_trend = float(second_half[-1] - second_half[0])
        if fh_trend > TREND_MIN and abs(sh_trend) < 0.1:
            return "SATURATING_RISE", stats
        if fh_trend < -TREND_MIN and abs(sh_trend) < 0.1:
            return "SATURATING_DECAY", stats

    # ---- step-like changes vs spikes/bursts ----
    max_step = float(np.max(np.abs(diffs)))
    avg_step = float(np.mean(np.abs(diffs)))
    if max_step > 0.8 and avg_step < 0.4:
        # Single big step then new plateau.
        if global_trend > 0.2:
            return "STEP_UP", stats
        elif global_trend < -0.2:
            return "STEP_DOWN", stats

    # Spiky behaviour: big excursions but not persistent plateau.
    if num_spike_pts > 0 and amp >= STABLE_AMP_EPS:
        contiguous = num_spike_pts <= max(2, L // 3)
        if contiguous and noise < NOISY_NOISE_EPS:
            if spike_frac >= 2.0 / 3.0:
                return "SINGLE_SPIKE_LATE", stats
            elif spike_frac <= 1.0 / 3.0:
                return "SINGLE_SPIKE_EARLY", stats
            else:
                return "SINGLE_SPIKE_MID", stats
        else:
            # Distinguish low-rate vs high-rate bursts by density.
            spike_density = num_spike_pts / L
            if spike_density < 0.3:
                return "LOW_RATE_BURSTS", stats
            else:
                return "HIGH_RATE_BURSTS", stats

    # ---- U / hump-shaped patterns ----
    idx_max = int(np.argmax(norm))
    idx_min = int(np.argmin(norm))
    if idx_min < idx_max and idx_min > 0 and idx_max < L - 1:
        # down then up
        if norm[idx_min] < -0.3 and norm[idx_max] > 0.3:
            return "U_SHAPED", stats
    if idx_max < idx_min and idx_max > 0 and idx_min < L - 1:
        # up then down
        if norm[idx_max] > 0.3 and norm[idx_min] < -0.3:
            return "HUMP_SHAPED", stats

    # ---- noisy / drifting ----
    if abs(global_trend) > 0.2 and noise >= STABLE_NOISE_EPS:
        return "DRIFT_WITH_NOISE", stats

    # Fallback.
    return "NOISY_FLUCTUATIONS", stats


def series_role_from_label(label: str) -> str:
    """
    Coarse role of an individual attribute within a cross-attribute patch.
    """
    stable_labels = {
        "ZERO_OR_NEAR_ZERO",
        "LOW_STABLE",
        "HIGH_STABLE",
    }
    trend_labels = {
        "SLOW_RISE",
        "SLOW_DECAY",
        "FAST_RISE",
        "FAST_DECAY",
        "SATURATING_RISE",
        "SATURATING_DECAY",
        "DRIFT_WITH_NOISE",
    }
    spiky_labels = {
        "SINGLE_SPIKE_EARLY",
        "SINGLE_SPIKE_MID",
        "SINGLE_SPIKE_LATE",
        "REPEATED_BURSTS",
        "LOW_RATE_BURSTS",
        "HIGH_RATE_BURSTS",
        "STEP_UP",
        "STEP_DOWN",
        "U_SHAPED",
        "HUMP_SHAPED",
    }

    if label in stable_labels:
        return "stable"
    if label in trend_labels:
        return "trend"
    if label in spiky_labels:
        return "spiky"
    if label == "UNUSED_PROTO":
        return "stable"
    return "noisy"


# =========================================================
# Cross-attribute labels / phrases and heuristics
# =========================================================

CROSS_LABEL_TO_PHRASE = {
    # housekeeping
    "UNUSED_PROTO": "prototype rarely activated in training windows",

    # all quiet
    "ALL_STABLE": "all monitored attributes stay stable in this patch",

    # workload / wear vs errors
    "WORKLOAD_SPIKE_NO_ERRORS":
        "workload or throughput counters spike but error counters stay flat",
    "WEAROUT_NO_ERRORS":
        "wear or usage attributes increase steadily while error counters stay near zero",
    "WEAROUT_WITH_ERRORS":
        "wear or usage attributes increase and one or more error counters also drift up",

    # error bursts
    "FEW_ERROR_BURSTS":
        "a few attributes show bursts while most remain stable",
    "MULTI_ERROR_BURSTS":
        "many attributes show bursts or spikes together",
    "ERROR_BURST_ISOLATED":
        "a short-lived burst in one or two error attributes with little change elsewhere",
    "ERROR_BURST_WIDESPREAD":
        "coordinated bursts across many error and reliability attributes",

    # mixed trends
    "MULTI_ATTR_TRENDS":
        "several attributes show gradual trends but no sharp bursts",
    "TREND_PLUS_BURSTS":
        "gradual trends accompanied by bursts on a subset of attributes",
    "MIXED_PATTERN":
        "a mix of stable, trending, and spiky attributes",

    # counter pathologies
    "RESET_OR_WRAP":
        "one or more counters drop sharply, suggesting a reset or wrap-around",
}


def _feature_group(name: str) -> str:
    """
    Very light-weight grouping of attributes based on their names.
    If your feature names are just 'r_5', 'r_187', you can later plug
    in a custom mapping here.
    """
    lname = name.lower()

    error_keywords = [
        "error", "err", "uncorr", "uncorrect",
        "realloc", "re-alloc", "reallocated",
        "pending", "crc", "timeout", "fail", "failure",
        "bad_block", "bad block", "media_error",
    ]
    wear_keywords = [
        "wear", "erase", "erases", "erase_count", "pe_cycle",
        "lifetime", "life", "program", "nand_wr", "nand writes",
    ]
    workload_keywords = [
        "read", "write", "throughput", "iops", "bandwidth", "util",
        "queue", "qdepth", "latency",
    ]

    if any(k in lname for k in error_keywords):
        return "error"
    if any(k in lname for k in wear_keywords):
        return "wear"
    if any(k in lname for k in workload_keywords):
        return "workload"
    return "other"


def classify_cross_pattern(
    patch_stack: np.ndarray,
    feature_names: List[str],
) -> Tuple[str, Dict]:
    """
    patch_stack: [N_k, F, L] for one cross prototype k.
    Aggregate per-feature patterns and summarise.
    """
    if patch_stack.size == 0:
        return "UNUSED_PROTO", {"num_patches": 0}

    mean_patch = patch_stack.mean(axis=0)  # [F, L]
    F, L = mean_patch.shape

    per_feature = []
    role_counts = {"stable": 0, "trend": 0, "spiky": 0, "noisy": 0}
    group_counts = {
        "error": {"stable": 0, "trend": 0, "spiky": 0, "noisy": 0},
        "wear": {"stable": 0, "trend": 0, "spiky": 0, "noisy": 0},
        "workload": {"stable": 0, "trend": 0, "spiky": 0, "noisy": 0},
        "other": {"stable": 0, "trend": 0, "spiky": 0, "noisy": 0},
    }
    reset_like = False

    for f_idx in range(F):
        label_f, stats_f = classify_attr_trend(mean_patch[f_idx])
        role = series_role_from_label(label_f)
        g = _feature_group(feature_names[f_idx])

        role_counts[role] += 1
        group_counts[g][role] += 1

        if label_f == "STEP_DOWN":
            reset_like = True

        per_feature.append(
            {
                "feature": feature_names[f_idx],
                "label": label_f,
                "role": role,
                "group": g,
                "stats": stats_f,
            }
        )

    # Global fractions.
    F_float = float(F)
    frac_stable = role_counts["stable"] / F_float
    frac_spiky = role_counts["spiky"] / F_float
    frac_trend = role_counts["trend"] / F_float

    # 1) All-stable pattern.
    if frac_stable > 0.8 and frac_spiky < 0.1 and frac_trend < 0.1:
        cross_label = "ALL_STABLE"
        stats = {
            "num_patches": int(patch_stack.shape[0]),
            "role_counts": role_counts,
            "group_counts": group_counts,
            "per_feature": per_feature,
        }
        return cross_label, stats

    # Helper for group fractions.
    def group_frac(group: str, role: str) -> float:
        num = sum(group_counts[group].values())
        if num == 0:
            return 0.0
        return group_counts[group][role] / float(num)

    # 2) Specialised group-based patterns.

    # Workload spike but errors quiet.
    err_num = sum(group_counts["error"].values())
    work_num = sum(group_counts["workload"].values())
    wear_num = sum(group_counts["wear"].values())

    if work_num > 0 and err_num > 0:
        wl_spiky = group_frac("workload", "spiky")
        wl_trend = group_frac("workload", "trend")
        err_spiky = group_frac("error", "spiky")
        err_trend = group_frac("error", "trend")
        if (wl_spiky + wl_trend) >= 0.3 and (err_spiky + err_trend) < 0.1:
            cross_label = "WORKLOAD_SPIKE_NO_ERRORS"
            stats = {
                "num_patches": int(patch_stack.shape[0]),
                "role_counts": role_counts,
                "group_counts": group_counts,
                "per_feature": per_feature,
            }
            return cross_label, stats

    # Wear-out vs errors.
    if wear_num > 0:
        wear_trend = group_frac("wear", "trend")
        err_spiky = group_frac("error", "spiky")
        err_trend = group_frac("error", "trend")
        if wear_trend >= 0.3 and (err_spiky + err_trend) < 0.1:
            cross_label = "WEAROUT_NO_ERRORS"
            stats = {
                "num_patches": int(patch_stack.shape[0]),
                "role_counts": role_counts,
                "group_counts": group_counts,
                "per_feature": per_feature,
            }
            return cross_label, stats
        if wear_trend >= 0.3 and (err_spiky + err_trend) >= 0.1:
            cross_label = "WEAROUT_WITH_ERRORS"
            stats = {
                "num_patches": int(patch_stack.shape[0]),
                "role_counts": role_counts,
                "group_counts": group_counts,
                "per_feature": per_feature,
            }
            return cross_label, stats

    # Error burst isolated / widespread.
    if err_num > 0:
        err_spiky = group_frac("error", "spiky")
        if 0.1 <= err_spiky < 0.4:
            cross_label = "ERROR_BURST_ISOLATED"
            stats = {
                "num_patches": int(patch_stack.shape[0]),
                "role_counts": role_counts,
                "group_counts": group_counts,
                "per_feature": per_feature,
            }
            return cross_label, stats
        if err_spiky >= 0.4:
            cross_label = "ERROR_BURST_WIDESPREAD"
            stats = {
                "num_patches": int(patch_stack.shape[0]),
                "role_counts": role_counts,
                "group_counts": group_counts,
                "per_feature": per_feature,
            }
            return cross_label, stats

    # 3) Reset-like patterns.
    if reset_like:
        cross_label = "RESET_OR_WRAP"
        stats = {
            "num_patches": int(patch_stack.shape[0]),
            "role_counts": role_counts,
            "group_counts": group_counts,
            "per_feature": per_feature,
        }
        return cross_label, stats

    # 4) Generic role-based patterns.
    if frac_spiky < 0.1 and frac_trend >= 0.2:
        cross_label = "MULTI_ATTR_TRENDS"
    elif frac_spiky >= 0.4 and frac_stable > 0.3:
        cross_label = "MULTI_ERROR_BURSTS"
    elif frac_spiky >= 0.2 and frac_stable > 0.5:
        cross_label = "FEW_ERROR_BURSTS"
    elif frac_trend >= 0.2 and frac_spiky >= 0.2:
        cross_label = "TREND_PLUS_BURSTS"
    else:
        cross_label = "MIXED_PATTERN"

    stats = {
        "num_patches": int(patch_stack.shape[0]),
        "role_counts": role_counts,
        "group_counts": group_counts,
        "per_feature": per_feature,
    }
    return cross_label, stats


# =========================================================
# Patch collection from train windows
# =========================================================

def collect_patches_for_prototypes(
    X_raw: np.ndarray,
    X_norm: np.ndarray,
    feature_names: List[str],
    attr_ae: AttrAutoencoder,
    cross_ae: CrossAutoencoder,
    attr_centers: np.ndarray,
    cross_centers: np.ndarray,
    patch_len_attr: int,
    patch_len_cross: int,
    attr_threshold: float,
    cross_threshold: float,
    device: torch.device,
) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
    """
    For each prototype k, collect raw patches that are well explained
    by that prototype (distance <= threshold).

    Returns:
      attr_buckets:  list of length K_attr, each is list of [L] arrays
      cross_buckets: list of length K_cross, each is list of [F, L] arrays
    """
    N, T, F = X_norm.shape
    K_attr = attr_centers.shape[0]
    K_cross = cross_centers.shape[0]

    attr_buckets: List[List[np.ndarray]] = [[] for _ in range(K_attr)]
    cross_buckets: List[List[np.ndarray]] = [[] for _ in range(K_cross)]

    # For reservoir sampling
    attr_seen = np.zeros(K_attr, dtype=np.int64)
    cross_seen = np.zeros(K_cross, dtype=np.int64)

    attr_ae.to(device).eval()
    cross_ae.to(device).eval()

    # --- progress iterator ---
    if tqdm is not None:
        iterator = tqdm(range(N), desc="Collecting patches", unit="win")
    else:
        iterator = range(N)

    for n in iterator:
        if tqdm is None and (n + 1) % 100000 == 0:
            frac = (n + 1) / N
            print(f"Processed {n + 1}/{N} windows ({frac:.1%})")

        win_norm = X_norm[n]  # [T, F]
        win_raw = X_raw[n]    # [T, F]

        # -----------------------------------------------------
        # 1) Attribute patches: vectorised over all features
        # -----------------------------------------------------
        # win_norm: [T, F] -> [F, T]
        win_norm_FT = torch.as_tensor(win_norm.T, dtype=torch.float32, device=device)  # [F, T]
        patches_attr = win_norm_FT.unfold(1, patch_len_attr, patch_len_attr)           # [F, P, L]
        F_, P_attr, L_attr = patches_attr.shape

        if P_attr > 0:
            patches_attr = patches_attr.reshape(F_ * P_attr, 1, L_attr)  # [F*P, 1, L]

            with torch.no_grad():
                emb = attr_ae.encode(patches_attr).cpu().numpy()  # [F*P, D]

            dists = np.linalg.norm(
                emb[:, None, :] - attr_centers[None, :, :],
                axis=-1,
            )  # [F*P, K_attr]
            proto_ids = dists.argmin(axis=1)
            min_dists = dists.min(axis=1)

            win_raw_FT = win_raw.T  # [F, T] (numpy)

            for idx_flat, (k_attr, d_min) in enumerate(zip(proto_ids, min_dists)):
                if d_min > attr_threshold:
                    continue

                f_idx = idx_flat // P_attr
                p_idx = idx_flat % P_attr
                start = p_idx * patch_len_attr
                end = start + patch_len_attr
                if end > T:
                    continue

                patch_raw = win_raw_FT[f_idx, start:end].astype(np.float32)
                k = int(k_attr)

                # reservoir sampling into attr_buckets[k]
                attr_seen[k] += 1
                if attr_seen[k] <= MAX_ATTR_PATCHES_PER_CLUSTER:
                    attr_buckets[k].append(patch_raw)
                else:
                    r = np.random.randint(0, attr_seen[k])
                    if r < MAX_ATTR_PATCHES_PER_CLUSTER:
                        attr_buckets[k][r] = patch_raw

        # -----------------------------------------------------
        # 2) Cross-attribute patches (same as before, but capped)
        # -----------------------------------------------------
        series_norm2 = win_norm  # [T, F]
        series_t2 = torch.as_tensor(series_norm2, dtype=torch.float32, device=device)

        patches2 = series_t2.unfold(0, patch_len_cross, patch_len_cross)  # [P, L, F]
        if patches2.numel() > 0:
            patches2 = patches2.permute(0, 2, 1).unsqueeze(1)  # [P, 1, F, L]
            with torch.no_grad():
                emb2 = cross_ae.encode(patches2).cpu().numpy()  # [P, D]
            dists2 = np.linalg.norm(
                emb2[:, None, :] - cross_centers[None, :, :],
                axis=-1,
            )  # [P, K_cross]
            proto_ids2 = dists2.argmin(axis=1)
            min_dists2 = dists2.min(axis=1)

            for p_idx, (k_cross, d_min) in enumerate(zip(proto_ids2, min_dists2)):
                if d_min > cross_threshold:
                    continue

                start = p_idx * patch_len_cross
                end = start + patch_len_cross
                if end > T:
                    continue

                patch_raw = win_raw[start:end, :].T.astype(np.float32)  # [F, L]
                k = int(k_cross)

                # reservoir sampling into cross_buckets[k]
                cross_seen[k] += 1
                if cross_seen[k] <= MAX_CROSS_PATCHES_PER_CLUSTER:
                    cross_buckets[k].append(patch_raw)
                else:
                    r = np.random.randint(0, cross_seen[k])
                    if r < MAX_CROSS_PATCHES_PER_CLUSTER:
                        cross_buckets[k][r] = patch_raw

    return attr_buckets, cross_buckets


# =========================================================
# Vocab construction
# =========================================================

def build_attr_vocab(attr_buckets: List[List[np.ndarray]]) -> Dict[int, Dict]:
    vocab: Dict[int, Dict] = {}
    for k, patches in enumerate(attr_buckets):
        if not patches:
            label = "UNUSED_PROTO"
            phrase = ATTR_LABEL_TO_PHRASE[label]
            vocab[k] = {
                "label": label,
                "phrase": phrase,
                "num_patches": 0,
                "stats": {},
            }
            continue

        stack = np.stack(patches, axis=0)  # [N_k, L]
        mean_series = stack.mean(axis=0)
        label, stats = classify_attr_trend(mean_series)
        phrase = ATTR_LABEL_TO_PHRASE.get(label, "unclassified pattern")

        vocab[k] = {
            "label": label,
            "phrase": phrase,
            "num_patches": int(stack.shape[0]),
            "stats": stats,
        }
    return vocab


def build_cross_vocab(
    cross_buckets: List[List[np.ndarray]],
    feature_names: List[str],
) -> Dict[int, Dict]:
    vocab: Dict[int, Dict] = {}
    for k, patches in enumerate(cross_buckets):
        if not patches:
            label = "UNUSED_PROTO"
            phrase = CROSS_LABEL_TO_PHRASE[label]
            vocab[k] = {
                "label": label,
                "phrase": phrase,
                "num_patches": 0,
                "stats": {},
            }
            continue

        stack = np.stack(patches, axis=0)  # [N_k, F, L]
        label, stats = classify_cross_pattern(stack, feature_names)
        phrase = CROSS_LABEL_TO_PHRASE.get(label, "unclassified cross-attribute pattern")

        vocab[k] = {
            "label": label,
            "phrase": phrase,
            "num_patches": int(stack.shape[0]),
            "stats": stats,
        }
    return vocab


# =========================================================
# Main
# =========================================================


def infer_artifact_prefix(artifact_root: Path) -> str:
    name = artifact_root.name.lower()
    if "mb1" in name:
        return "mb1"
    return "mb2"


def main():
    parser = argparse.ArgumentParser(
        description="Learn prototype phrases from saved SMARTTalk clusters and train windows.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device string, e.g. 'cuda:0' or 'cpu'.",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default="data/processed/MB2_round1/train.npz",
        help="Path to train.npz with raw windows.",
    )
    parser.add_argument(
        "--artifact-root",
        type=str,
        default="data/artifacts/MB2_round1",
        help="Folder that contains <prefix>_prototypes.npz, <prefix>_scaler.npz, and encoder .pt files.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=800000,
        help=(
            "Maximum number of windows to use for phrase learning. "
            "If N > max_windows, a random subset is sampled. "
            "Set <=0 to disable subsampling."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write JSON vocab (default: <artifact-root>/mb2_vocab.json).",
    )
    parser.add_argument(
        "--output-npz",
        type=str,
        default=None,
        help=(
            "Optional path to write updated prototype npz with phrases "
            "(default: <artifact-root>/mb2_prototypes_with_phrases.npz)."
        ),
    )
    parser.add_argument(
        "--artifact-prefix",
        type=str,
        default=None,
        help="Optional artifact filename prefix, e.g. mb1 or mb2. Default: infer from artifact-root.",
    )

    args = parser.parse_args()
    device = get_device(args.device)

    artifact_root = Path(args.artifact_root)
    artifact_prefix = args.artifact_prefix or infer_artifact_prefix(artifact_root)
    train_path = Path(args.train_path)
    scaler_path = artifact_root / f"{artifact_prefix}_scaler.npz"
    proto_path = artifact_root / f"{artifact_prefix}_prototypes.npz"
    attr_encoder_pt = artifact_root / f"{artifact_prefix}_attr_autoencoder.pt"
    cross_encoder_pt = artifact_root / f"{artifact_prefix}_cross_autoencoder.pt"

    if args.output_json is None:
        output_json = artifact_root / f"{artifact_prefix}_vocab.json"
    else:
        output_json = Path(args.output_json)

    if args.output_npz is None:
        output_npz = artifact_root / f"{artifact_prefix}_prototypes_with_phrases.npz"
    else:
        output_npz = Path(args.output_npz)

    print(f"Loading train windows from {train_path} ...")
    train_npz = np.load(train_path)
    X_train = train_npz["X"].astype(np.float32)          # [N, T, F_train]
    train_features = train_npz["features"].tolist()
    N_full, T, F_train = X_train.shape
    print(f"Train windows: N={N_full}, T={T}, F={F_train}")

    print(f"Loading scaler from {scaler_path} ...")
    scaler = np.load(scaler_path)
    feat_mean = scaler["mean"].astype(np.float32)        # [1,1,F]
    feat_std = scaler["std"].astype(np.float32)
    scaler_features = scaler["features"].tolist()
    F_scaler = len(scaler_features)
    print(f"Scaler features: F={F_scaler}")

    # Reorder train features to match scaler/prototype feature order.
    if train_features != scaler_features:
        print("Reordering train features to match scaler feature order...")
        name_to_idx = {name: i for i, name in enumerate(train_features)}
        idxs = [name_to_idx[name] for name in scaler_features]
        X_train = X_train[:, :, idxs]
    feature_names = scaler_features
    assert X_train.shape[2] == F_scaler, "Feature mismatch after reordering."

    # Standardise train windows using the same scaler as MB2.
    X_train_norm = (X_train - feat_mean) / feat_std

    # -------- subsample windows for phrase learning --------
    if args.max_windows is not None and args.max_windows > 0 and args.max_windows < N_full:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(N_full, size=args.max_windows, replace=False)
        X_train = X_train[idx]
        X_train_norm = X_train_norm[idx]
        N = X_train.shape[0]
        print(f"Subsampling windows for phrase learning: {N} / {N_full}")
    else:
        N = N_full
        print(f"Using all windows for phrase learning: N={N}")

    print(f"Loading prototypes from {proto_path} ...")
    proto = np.load(proto_path, allow_pickle=True)

    def scalar(name: str) -> float:
        return float(np.atleast_1d(proto[name])[0])

    attr_centers = proto["attr_centers"]
    cross_centers = proto["cross_centers"]
    attr_threshold = scalar("attr_threshold")
    cross_threshold = scalar("cross_threshold")
    patch_len_attr = int(scalar("patch_len_attr"))
    patch_len_cross = int(scalar("patch_len_cross"))
    attr_emb_dim = int(scalar("attr_emb_dim"))
    cross_emb_dim = int(scalar("cross_emb_dim"))
    n_attr_clusters = int(scalar("n_attr_clusters"))
    n_cross_clusters = int(scalar("n_cross_clusters"))

    print(f"Attr clusters: {n_attr_clusters}, Cross clusters: {n_cross_clusters}")
    print(f"Patch lens: attr={patch_len_attr}, cross={patch_len_cross}")
    print(f"Thresholds: attr={attr_threshold:.4f}, cross={cross_threshold:.4f}")

    # Instantiate encoders and load weights.
    print(f"Loading encoders from {attr_encoder_pt} and {cross_encoder_pt} ...")
    attr_ae = AttrAutoencoder(patch_len=patch_len_attr, emb_dim=attr_emb_dim)
    attr_ae.load_state_dict(torch.load(attr_encoder_pt, map_location=device))

    cross_ae = CrossAutoencoder(
        num_features=len(feature_names),
        patch_len=patch_len_cross,
        emb_dim=cross_emb_dim,
    )
    cross_ae.load_state_dict(torch.load(cross_encoder_pt, map_location=device))

    # Collect representative patches per prototype.
    print("Collecting representative patches for each prototype...")
    attr_buckets, cross_buckets = collect_patches_for_prototypes(
        X_raw=X_train,
        X_norm=X_train_norm,
        feature_names=feature_names,
        attr_ae=attr_ae,
        cross_ae=cross_ae,
        attr_centers=attr_centers,
        cross_centers=cross_centers,
        patch_len_attr=patch_len_attr,
        patch_len_cross=patch_len_cross,
        attr_threshold=attr_threshold,
        cross_threshold=cross_threshold,
        device=device,
    )

    # Safety check: we expect K buckets.
    assert len(attr_buckets) == n_attr_clusters
    assert len(cross_buckets) == n_cross_clusters

    print("Building attribute vocab...")
    attr_vocab = build_attr_vocab(attr_buckets)

    print("Building cross-attribute vocab...")
    cross_vocab = build_cross_vocab(cross_buckets, feature_names)

    # Save JSON for easy inspection / manual editing.
    vocab_payload = {
        "meta": {
            "patch_len_attr": patch_len_attr,
            "patch_len_cross": patch_len_cross,
            "feature_names": feature_names,
            "attr_threshold": attr_threshold,
            "cross_threshold": cross_threshold,
        },
        "attr": attr_vocab,
        "cross": cross_vocab,
    }
    print(f"Writing vocab JSON to {output_json} ...")
    with open(output_json, "w") as f:
        json.dump(vocab_payload, f, indent=2)

    # Also save an updated prototype memory npz that includes phrases.
    print(f"Writing updated prototypes (with phrases) to {output_npz} ...")
    proto_data = {k: proto[k] for k in proto.files}
    proto_data["attr_labels"] = np.array(
        [attr_vocab[k]["label"] for k in range(n_attr_clusters)],
        dtype=object,
    )
    proto_data["attr_phrases"] = np.array(
        [attr_vocab[k]["phrase"] for k in range(n_attr_clusters)],
        dtype=object,
    )
    proto_data["cross_labels"] = np.array(
        [cross_vocab[k]["label"] for k in range(n_cross_clusters)],
        dtype=object,
    )
    proto_data["cross_phrases"] = np.array(
        [cross_vocab[k]["phrase"] for k in range(n_cross_clusters)],
        dtype=object,
    )

    np.savez_compressed(output_npz, **proto_data)
    print("Done. PrototypeMemory now has labels + phrases for each cluster.")


if __name__ == "__main__":
    main()
