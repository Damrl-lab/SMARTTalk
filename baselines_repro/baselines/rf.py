"""Random Forest baseline (Alter et al., "SSD Failures in the Field", SC'19).

Per-attribute statistical features over the SMART window, the majority class
downsampled to 1:1 for training, and a Random Forest classifier. Evaluated on
the fixed 1:23 imbalanced test set pooled across the three temporal rounds.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import pooled_train, downsample, per_attribute_stats, cli  # noqa: E402

CONFIG = {
    "MB1": {"n_estimators": 300, "max_depth": 2},
    "MB2": {"n_estimators": 300, "max_depth": 2},
}


class RandomForestBaseline:
    def __init__(self, n_estimators=300, max_depth=None, seed=0):
        self.n_estimators = n_estimators; self.max_depth = max_depth; self.seed = seed

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(per_attribute_stats(X))
        self.clf = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                          n_jobs=-1, random_state=self.seed)
        self.clf.fit(self.scaler.transform(per_attribute_stats(X)), y)
        return self

    def score(self, X):
        return self.clf.predict_proba(self.scaler.transform(per_attribute_stats(X)))[:, 1]


def train(data_root, dataset, seed=0):
    X, y = pooled_train(data_root, dataset)
    idx = downsample(y, neg_per_pos=1.0, seed=seed)          # 1:1 majority downsampling
    return RandomForestBaseline(**CONFIG[dataset], seed=seed).fit(X[idx], y[idx])


if __name__ == "__main__":
    cli("RF", train)
