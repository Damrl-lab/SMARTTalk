"""Train the baselines from the data splits and save their checkpoints.

    python train_baselines.py                              # all baselines, MB1 and MB2
    python train_baselines.py --models RF,NN --datasets MB1
    python train_baselines.py --out checkpoints_retrained  # keep the shipped ones intact

After training, rebuild the table on the trained checkpoints with:
    python reproduce_table5.py --checkpoints <out>
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "baselines"))
from common import train_and_save  # noqa: E402

MODELS = ["RF", "NN", "EC", "AE", "LSTM", "MVTRF", "MSFRD"]
DATASETS = ["MB1", "MB2"]
MODULE = {"RF": "rf", "NN": "nn", "EC": "ec", "AE": "ae", "LSTM": "lstm", "MVTRF": "mvtrf", "MSFRD": "msfrd"}


def main():
    ap = argparse.ArgumentParser(description="Train SMART baselines and save checkpoints")
    ap.add_argument("--data-root", default="data/splits_sampled")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        mod = importlib.import_module(MODULE[name])
        for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
            train_and_save(mod.train, name, ds, args.data_root, Path(args.out) / f"{name}_{ds}.joblib")


if __name__ == "__main__":
    main()
