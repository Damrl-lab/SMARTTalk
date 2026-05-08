"""Wrappers around table-generation and explanation-evaluation scripts."""

from __future__ import annotations

import shutil
from pathlib import Path

from smarttalk.common.runner import run_python
from smarttalk.common.paths import CONFIG_DIR, LEGACY_DIR, RESULTS_DIR, ROOT


def make_table5_status(*args: str) -> None:
    run_python(LEGACY_DIR / "scripts" / "generate_status_sampled_1to23.py", args=args, cwd=ROOT)


def make_paper_tables(*args: str) -> None:
    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    paper_tables = CONFIG_DIR / "paper_tables"

    shutil.copy2(paper_tables / "table5_status.csv", tables_dir / "table5_status.csv")
    shutil.copy2(paper_tables / "table6_ttf.csv", tables_dir / "table6_ttf.csv")
    shutil.copy2(paper_tables / "table7_explanations.csv", tables_dir / "table7_explanations.csv")

    # Keep the legacy compatibility aliases fresh as well.
    legacy_tables_dir = RESULTS_DIR / "paper_tables"
    if legacy_tables_dir.exists():
        try:
            shutil.copy2(paper_tables / "table5_status.csv", legacy_tables_dir / "table5_status.csv")
            shutil.copy2(paper_tables / "table6_ttf.csv", legacy_tables_dir / "table6_ttf.csv")
            shutil.copy2(paper_tables / "table7_explanations.csv", legacy_tables_dir / "table7_explanations.csv")
        except IsADirectoryError:
            pass


def run_table7_pipeline(*args: str) -> None:
    run_python(LEGACY_DIR / "scripts" / "run_table7_pipeline.py", args=args, cwd=ROOT)
