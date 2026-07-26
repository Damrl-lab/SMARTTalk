"""Rebuild the Table 5 baseline block for MB1 and MB2.

Loads each baseline checkpoint and its fixed 1:23 evaluation set, computes
precision / recall / F0.5 / FPR / FNR (pooled across rounds 1-3), and writes the
table.

    python reproduce_table5.py --checkpoints checkpoints --eval-sets eval_sets --out results
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import joblib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "baselines"))
from common import evaluate_on_set  # noqa: E402
import rf, nn, ec, ae, lstm, mvtrf, msfrd  # noqa: E402,F401  (register checkpoint classes)

MODELS = ["RF", "NN", "EC", "AE", "LSTM", "MVTRF", "MSFRD"]
DATASETS = ["MB1", "MB2"]


def main():
    ap = argparse.ArgumentParser(description="Rebuild the Table 5 baseline block")
    ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--eval-sets", default="eval_sets")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in MODELS:
        row = {"method": name}
        for ds in DATASETS:
            model = joblib.load(Path(args.checkpoints) / f"{name}_{ds}.joblib")
            m = evaluate_on_set(model, Path(args.eval_sets) / f"{name}_{ds}_test.npz")
            row.update({f"{ds}_P": round(m.precision, 2), f"{ds}_R": round(m.recall, 2),
                        f"{ds}_F05": round(m.f05, 2), f"{ds}_FPR": round(m.fpr, 4),
                        f"{ds}_FNR": round(m.fnr, 2)})
            print(f"{name:6s} {ds}: P={m.precision:.2f} R={m.recall:.2f} F0.5={m.f05:.2f} "
                  f"FPR={m.fpr:.4f} FNR={m.fnr:.2f}")
        rows.append(row)

    cols = ["method"] + [f"{ds}_{k}" for ds in DATASETS for k in ["P", "R", "F05", "FPR", "FNR"]]
    with open(out / "table5_reproduced.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out / 'table5_reproduced.csv'}")


if __name__ == "__main__":
    main()
