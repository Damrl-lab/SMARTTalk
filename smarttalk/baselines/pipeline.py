"""Thin baseline dispatch helpers."""

from __future__ import annotations

from pathlib import Path

from smarttalk.common.runner import run_python
from smarttalk.common.paths import LEGACY_DIR, ROOT


BASELINE_SCRIPTS = {
    "rf": "rf_nn.py",
    "nn": "rf_nn.py",
    "ec": "ec.py",
    "ae": "ae.py",
    "lstm": "lstm.py",
    "mvtrf": "mvtrf.py",
    "msfrd": "msfrd.py",
}


def run_baseline(model: str, *args: str) -> None:
    script_name = BASELINE_SCRIPTS[model]
    run_python(LEGACY_DIR / "baselines" / script_name, args=args, cwd=ROOT)
