"""Subprocess helpers for thin wrapper scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable

from .logging_utils import log_step
from .paths import ROOT


def run_python(script: Path, args: Iterable[str] = (), cwd: Path | None = None) -> None:
    cmd = [sys.executable, str(script), *list(args)]
    log_step(" ".join(cmd))
    subprocess.run(cmd, cwd=cwd or ROOT, check=True)


def run_shell(command: list[str], cwd: Path | None = None) -> None:
    log_step(" ".join(command))
    subprocess.run(command, cwd=cwd or ROOT, check=True)
