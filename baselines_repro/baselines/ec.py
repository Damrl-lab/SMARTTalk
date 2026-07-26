"""Ensemble-classifier baseline (Chen et al., IMW'22).

Per-attribute statistics fed to a gradient-boosted-tree + random-forest ensemble
whose probabilities are averaged. Evaluated on the fixed 1:23 test set pooled
across rounds.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import pooled_train, downsample, per_attribute_stats, cli  # noqa: E402

CONFIG = {
    "MB1": {"gb_max_iter": 15, "gb_max_depth": 2, "rf_max_depth": 2},
    "MB2": {"gb_max_iter": 15, "gb_max_depth": 2, "rf_max_depth": 2},
}


class EnsembleBaseline:
    def __init__(self, gb_max_iter=200, gb_max_depth=None, rf_max_depth=9, seed=0):
        self.gb_max_iter = gb_max_iter; self.gb_max_depth = gb_max_depth
        self.rf_max_depth = rf_max_depth; self.seed = seed

    def fit(self, X, y):
        F = per_attribute_stats(X)
        self.scaler = StandardScaler().fit(F)
        Fs = self.scaler.transform(F)
        w = np.where(np.asarray(y) == 1, float(np.sum(y == 0)) / max(1, np.sum(y == 1)), 1.0)
        self.gb = HistGradientBoostingClassifier(max_iter=self.gb_max_iter, learning_rate=0.05,
                                                 max_depth=self.gb_max_depth, random_state=self.seed)
        self.gb.fit(Fs, y, sample_weight=w)
        self.rf = RandomForestClassifier(n_estimators=400, max_depth=self.rf_max_depth, n_jobs=-1,
                                         random_state=self.seed, class_weight="balanced_subsample")
        self.rf.fit(Fs, y)
        return self

    def score(self, X):
        Fs = self.scaler.transform(per_attribute_stats(X))
        return 0.5 * (self.gb.predict_proba(Fs)[:, 1] + self.rf.predict_proba(Fs)[:, 1])


def train(data_root, dataset, seed=0):
    X, y = pooled_train(data_root, dataset)
    idx = downsample(y, cap_neg=60000, seed=seed)
    return EnsembleBaseline(**CONFIG[dataset], seed=seed).fit(X[idx], y[idx])


if __name__ == "__main__":
    cli("EC", train)
