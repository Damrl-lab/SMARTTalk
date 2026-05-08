"""Thin wrappers around the working data-preparation implementation."""

from __future__ import annotations

from pathlib import Path

from smarttalk.common.runner import run_python
from smarttalk.common.paths import LEGACY_DIR, ROOT


def run_preprocess_raw_logs(*args: str) -> None:
    run_python(LEGACY_DIR / "core" / "filter_dataset.py", args=args, cwd=ROOT)


def run_make_temporal_splits(*args: str) -> None:
    run_python(LEGACY_DIR / "scripts" / "build_processed_splits.py", args=args, cwd=ROOT)


def run_make_sampled_test(*args: str) -> None:
    run_python(LEGACY_DIR / "scripts" / "build_sampled_test_set.py", args=args, cwd=ROOT)
