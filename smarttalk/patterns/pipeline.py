"""Offline pattern-memory pipeline wrappers."""

from __future__ import annotations

from smarttalk.common.runner import run_python
from smarttalk.common.paths import LEGACY_DIR, ROOT


def run_offline_pipeline(*args: str) -> None:
    run_python(LEGACY_DIR / "scripts" / "build_offline_artifacts.py", args=args, cwd=ROOT)


def run_generate_phrase_dictionary(*args: str) -> None:
    run_python(LEGACY_DIR / "scripts" / "export_phrase_dictionary_stats.py", args=args, cwd=ROOT)
