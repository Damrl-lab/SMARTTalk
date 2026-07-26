"""Multi-view time-series random forest baseline (Zhang et al., FAST'23).

Three views of each SMART window -- the latest-day raw values, per-attribute
normalized histograms, and per-segment sequence statistics (coefficient of
variation, kurtosis, slope) -- plus their concatenation, each fed to its own
random forest; the four forests vote equally. Evaluated on the fixed 1:23 test
set pooled across rounds.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import pooled_train, downsample, cli  # noqa: E402

CONFIG = {"MB1": {"max_depth": 2}, "MB2": {"max_depth": 2}}


class MVTRFBaseline:
    def __init__(self, M=100, G=4, trees_per_view=100, max_depth=None, seed=0):
        self.M = M; self.G = G; self.tpv = trees_per_view; self.max_depth = max_depth; self.seed = seed

    def _hist(self, X):
        N, T, F = X.shape
        Xc = np.nan_to_num(X, nan=0.0).astype(np.float32)
        out = np.zeros((N, F * self.M), np.float32); row = np.arange(N)[:, None] * self.M
        for a in range(F):
            b = np.clip(np.digitize(Xc[:, :, a], self.edges[a][1:-1]), 0, self.M - 1)
            cnt = np.bincount((b + row).ravel(), minlength=N * self.M).astype(np.float32).reshape(N, self.M)
            s = cnt.sum(1, keepdims=True); s[s == 0] = 1.0
            out[:, a * self.M:(a + 1) * self.M] = cnt / s
        return out

    def _seq(self, X):
        N, T, F = X.shape
        Xc = np.nan_to_num(X, nan=0.0).astype(np.float64)
        feats = []
        for idx in np.array_split(np.arange(T), self.G):
            s = Xc[:, idx, :]; mean = s.mean(1); std = s.std(1)
            cvar = np.where(np.abs(mean) > 1e-6, std / (mean + 1e-9), 0.0)
            dev = s - mean[:, None, :]; m2 = np.mean(dev ** 2, 1) + 1e-9; m4 = np.mean(dev ** 4, 1)
            kurt = np.clip(m4 / m2 ** 2 - 3.0, -1e6, 1e6)
            slope = (s[:, -1, :] - s[:, 0, :]) / max(1, len(idx) - 1)
            feats += [cvar, kurt, slope]
        return np.nan_to_num(np.concatenate(feats, 1)).astype(np.float32)

    def _views(self, X):
        raw = np.nan_to_num(X[:, -1, :]).astype(np.float32); hist = self._hist(X); seq = self._seq(X)
        return {"raw": raw, "hist": hist, "seq": seq, "comb": np.concatenate([raw, hist, seq], 1)}

    def fit(self, X, y):
        Xc = np.nan_to_num(X, nan=0.0); self.edges = []
        for a in range(X.shape[2]):
            lo, hi = float(Xc[:, :, a].min()), float(Xc[:, :, a].max())
            self.edges.append(np.linspace(lo, hi if hi > lo else lo + 1.0, self.M + 1))
        self.forests = {}
        for v, Xv in self._views(X).items():
            rf = RandomForestClassifier(n_estimators=self.tpv, max_depth=self.max_depth, n_jobs=-1,
                                        random_state=self.seed, class_weight="balanced_subsample")
            self.forests[v] = rf.fit(Xv, y)
        return self

    def score(self, X):
        v = self._views(X)
        return np.mean([self.forests[k].predict_proba(v[k])[:, 1] for k in self.forests], axis=0)


def train(data_root, dataset, seed=0):
    X, y = pooled_train(data_root, dataset)
    idx = downsample(y, cap_neg=60000, seed=seed)
    return MVTRFBaseline(**CONFIG[dataset], seed=seed).fit(X[idx], y[idx])


if __name__ == "__main__":
    cli("MVTRF", train)
