#!/usr/bin/env python3
"""Downsample processed SMART window splits to a fixed failed:healthy ratio,
memory-safely, without decompressing the whole array into RAM.

For each split .npz (train / val / test) it keeps ALL failed windows and a random
sample of healthy windows at the requested ratio (default 1:23), streaming the X
array row-by-row out of the compressed archive so peak memory stays near the
output size (a few GB) rather than the full decompressed size (tens of GB).

Single file:
    python sample_splits.py --in  data/splits/MB1_round1/train.npz \
                            --out data/splits_sampled/MB1_round1/train.npz \
                            --ratio 23 --seed 2026

Whole split tree (loops MB1/MB2 x rounds x train/val/test):
    python sample_splits.py --splits-root data/splits \
                            --out-root    data/splits_sampled \
                            --ratio 23 --seed 2026
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib.format import read_magic, read_array_header_1_0, read_array_header_2_0

CHUNK_ROWS = 8192


def _load_small(zf, name):
    with zf.open(name) as f:
        return np.lib.format.read_array(f)


def _select_indices(y, ratio, seed, max_rows=None):
    """Keep ALL failed windows + a random sample of healthy at the given ratio.
    max_rows (optional) only caps the healthy count so the total stays bounded;
    every failed window is always kept."""
    y = np.asarray(y)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    n_neg = min(len(neg), int(round(len(pos) * ratio)))
    if max_rows is not None:
        n_neg = min(n_neg, max(0, max_rows - len(pos)))
    neg_sel = rng.choice(neg, size=n_neg, replace=False) if n_neg < len(neg) else neg
    keep = np.concatenate([pos, neg_sel])
    keep.sort()
    return keep


def sample_file(in_path, out_path, ratio=23.0, seed=2026, max_rows=None):
    in_path, out_path = Path(in_path), Path(out_path)
    with zipfile.ZipFile(in_path) as zf:
        names = zf.namelist()
        # small arrays load fully (y, ttf, features are cheap)
        y = _load_small(zf, "y.npy")
        ttf = _load_small(zf, "ttf.npy") if "ttf.npy" in names else None
        features = _load_small(zf, "features.npy") if "features.npy" in names else None

        keep = _select_indices(y, ratio, seed, max_rows)
        keep_set_mask = np.zeros(len(y), dtype=bool)
        keep_set_mask[keep] = True

        # stream X.npy row-by-row, collecting only kept rows
        with zf.open("X.npy") as f:
            version = read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = read_array_header_1_0(f)
            elif version == (2, 0):
                shape, fortran, dtype = read_array_header_2_0(f)
            else:
                raise ValueError(f"unsupported .npy version {version}")
            if fortran:
                raise ValueError("Fortran-ordered X.npy is not supported by this streaming reader")
            n_rows = shape[0]
            row_shape = shape[1:]
            row_items = int(np.prod(row_shape)) if row_shape else 1
            row_bytes = row_items * dtype.itemsize

            out = np.empty((len(keep),) + row_shape, dtype=dtype)
            w = 0          # write pointer into out
            r = 0          # global row pointer
            while r < n_rows:
                n = min(CHUNK_ROWS, n_rows - r)
                raw = f.read(n * row_bytes)
                if len(raw) < n * row_bytes:
                    raise IOError("unexpected end of X.npy stream")
                block = np.frombuffer(raw, dtype=dtype).reshape((n,) + row_shape)
                m = keep_set_mask[r:r + n]
                sel = block[m]
                out[w:w + len(sel)] = sel
                w += len(sel); r += n
            assert w == len(keep), (w, len(keep))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save = {"X": out, "y": y[keep]}
    if ttf is not None:
        save["ttf"] = ttf[keep]
    if features is not None:
        save["features"] = features
    np.savez_compressed(out_path, **save)
    n_pos = int((y[keep] == 1).sum()); n_neg = int((y[keep] == 0).sum())
    print(f"{in_path.name}: kept {len(keep)} windows "
          f"(failed {n_pos}, healthy {n_neg}, ratio 1:{n_neg / max(1, n_pos):.1f}) -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", default=None)
    ap.add_argument("--out", dest="out_path", default=None)
    ap.add_argument("--splits-root", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--datasets", default="MB1,MB2")
    ap.add_argument("--rounds", default="1,2,3")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--ratio", type=float, default=23.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-rows", type=int, default=None, help="optional hard cap on output rows")
    args = ap.parse_args()

    if args.in_path and args.out_path:
        sample_file(args.in_path, args.out_path, args.ratio, args.seed, args.max_rows)
        return

    if not (args.splits_root and args.out_root):
        ap.error("give either --in/--out or --splits-root/--out-root")

    ds_idx = {"MB1": 0, "MB2": 1}
    split_idx = {"train": 0, "val": 1, "test": 2}
    root, out_root = Path(args.splits_root), Path(args.out_root)
    for ds in args.datasets.split(","):
        for rnd in args.rounds.split(","):
            for split in args.splits.split(","):
                src = root / f"{ds}_round{rnd}" / f"{split}.npz"
                if not src.exists():
                    print(f"skip (missing): {src}"); continue
                dst = out_root / f"{ds}_round{rnd}" / f"{split}.npz"
                # deterministic per-(dataset, round, split) seed so runs are reproducible
                seed = args.seed + ds_idx.get(ds, 9) * 1000 + int(rnd) * 10 + split_idx.get(split, 9)
                sample_file(src, dst, args.ratio, seed, args.max_rows)


if __name__ == "__main__":
    main()
