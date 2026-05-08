"""Validation helpers for processed SMART split files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


REQUIRED_KEYS = ("X", "y", "ttf", "features")


def summarize_npz_split(path: str | Path) -> Dict[str, object]:
    data = np.load(Path(path))
    missing = [key for key in REQUIRED_KEYS if key not in data.files]
    if missing:
        raise ValueError(f"Missing required keys in {path}: {missing}")
    x = data["X"]
    y = data["y"]
    ttf = data["ttf"]
    features = data["features"]
    if x.ndim != 3:
        raise ValueError(f"Expected X to be rank-3, got shape {x.shape}")
    if y.ndim != 1 or ttf.ndim != 1:
        raise ValueError("Expected y and ttf to be rank-1 arrays.")
    if len(y) != x.shape[0] or len(ttf) != x.shape[0]:
        raise ValueError("Window count mismatch between X, y, and ttf.")
    if len(features) != x.shape[2]:
        raise ValueError("Feature count mismatch between X and features.")
    return {
        "num_windows": int(x.shape[0]),
        "window_days": int(x.shape[1]),
        "num_attributes": int(x.shape[2]),
        "num_failed": int((y == 1).sum()),
        "num_healthy": int((y == 0).sum()),
    }


def validate_npz_split(path: str | Path) -> Tuple[bool, Dict[str, object]]:
    summary = summarize_npz_split(path)
    return True, summary
