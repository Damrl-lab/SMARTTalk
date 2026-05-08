#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FINALIZED_ROOT = ROOT.parent / "Finalized_Package"
OUTPUT_ROOT = ROOT / "results"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [sys.executable, "scripts/build_sampled_test_set.py"],
        cwd=FINALIZED_ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, "scripts/generate_status_sampled_1to23.py"],
        cwd=FINALIZED_ROOT,
        check=True,
    )

    sampled_root = FINALIZED_ROOT / "results" / "status_sampled_1to23"
    paper_root = FINALIZED_ROOT / "results" / "paper_tables"
    for source_root, names in [
        (
            sampled_root,
            [
                "confusion_matrices.csv",
                "status_metrics_with_fpr_fnr.csv",
                "status_table_with_fpr_fnr.csv",
                "audit.json",
            ],
        ),
        (
            paper_root,
            [
                "table5_status_with_fpr_fnr.csv",
                "table5_status_with_fpr_fnr_details.csv",
                "table5_status_with_fpr_fnr_audit.json",
            ],
        ),
    ]:
        for name in names:
            shutil.copy2(source_root / name, OUTPUT_ROOT / name)

    print(f"Wrote sampled 1:23 Table 5 artifacts to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
