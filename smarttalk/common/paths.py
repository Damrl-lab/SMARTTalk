"""Path helpers rooted at the final artifact package."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper"
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
SAMPLE_DATA_DIR = DATA_DIR / "sample_data"
ARTIFACT_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"
LEGACY_DIR = ROOT / "smarttalk" / "_legacy"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
