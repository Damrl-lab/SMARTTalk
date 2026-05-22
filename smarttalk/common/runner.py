"""Subprocess helpers for thin wrapper scripts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from .logging_utils import log_step
from .paths import ROOT


def _base_env() -> dict[str, str]:
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    root_str = str(ROOT)
    if existing_pythonpath:
        pythonpath_entries = existing_pythonpath.split(os.pathsep)
        if root_str not in pythonpath_entries:
            env["PYTHONPATH"] = os.pathsep.join([root_str, *pythonpath_entries])
    else:
        env["PYTHONPATH"] = root_str
    if "MPLCONFIGDIR" not in env:
        mpl_dir = ROOT / ".cache" / "matplotlib"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        env["MPLCONFIGDIR"] = str(mpl_dir)
    return env


def run_python(script: Path, args: Iterable[str] = (), cwd: Path | None = None) -> None:
    cmd = [sys.executable, str(script), *list(args)]
    log_step(" ".join(cmd))
    subprocess.run(cmd, cwd=cwd or ROOT, check=True, env=_base_env())


def run_shell(command: list[str], cwd: Path | None = None) -> None:
    log_step(" ".join(command))
    subprocess.run(command, cwd=cwd or ROOT, check=True, env=_base_env())
