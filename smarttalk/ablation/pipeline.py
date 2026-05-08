"""Wrappers around the sensitivity-study scripts."""

from __future__ import annotations

from smarttalk.common.runner import run_python
from smarttalk.common.paths import LEGACY_DIR, ROOT


def prepare_ablation_data(*args: str) -> None:
    run_python(LEGACY_DIR / "ablation" / "prepare_sensitivity_artifacts.py", args=args, cwd=ROOT)


def run_sensitivity_study(*args: str) -> None:
    run_python(LEGACY_DIR / "ablation" / "run_sensitivity_study.py", args=args, cwd=ROOT)


def run_ablation_bundle(*args: str) -> None:
    run_python(LEGACY_DIR / "ablation" / "run_sensitivity_bundle.py", args=args, cwd=ROOT)


def make_ablation_figures(*args: str) -> None:
    run_python(LEGACY_DIR / "ablation" / "ablation_readable_figures.py", args=args, cwd=ROOT)
