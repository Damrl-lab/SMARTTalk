"""Inference wrappers for SMARTTalk and its LLM baselines."""

from __future__ import annotations

from smarttalk.common.runner import run_python
from smarttalk.common.paths import LEGACY_DIR, ROOT


def run_raw_llm(*args: str) -> None:
    run_python(LEGACY_DIR / "core" / "raw_llm_eval.py", args=args, cwd=ROOT)


def run_heuristic_llm(*args: str) -> None:
    run_python(LEGACY_DIR / "core" / "heuristic_llm_eval.py", args=args, cwd=ROOT)


def run_smarttalk(*args: str) -> None:
    run_python(LEGACY_DIR / "core" / "llm_eval.py", args=args, cwd=ROOT)


def run_construct_summaries(*args: str) -> None:
    run_python(LEGACY_DIR / "core" / "mb2_dump_summaries.py", args=args, cwd=ROOT)
