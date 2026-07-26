"""Autoencoder baseline (Chakraborttii & Litz, SoCC'20).

A 1-class dense autoencoder (50-25-25-50, tanh) trained on healthy windows only;
per-window reconstruction error is the anomaly score. Evaluated on the fixed
1:23 test set pooled across rounds.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import pooled_train, flatten, MinMaxScaler, cli  # noqa: E402

CONFIG = {"MB1": {}, "MB2": {}}


class AutoencoderBaseline:
    def __init__(self, hidden=(50, 25, 25, 50), epochs=100, seed=0):
        self.hidden = hidden; self.epochs = epochs; self.seed = seed

    def fit(self, X, y):
        self.scaler = MinMaxScaler().fit(flatten(X))
        healthy = self.scaler.transform(flatten(X)[np.asarray(y) == 0])
        self.ae = MLPRegressor(hidden_layer_sizes=self.hidden, activation="tanh", solver="adam",
                               max_iter=self.epochs, n_iter_no_change=5, random_state=self.seed)
        self.ae.fit(healthy, healthy)
        return self

    def score(self, X):
        Xf = self.scaler.transform(flatten(X))
        return np.mean((self.ae.predict(Xf) - Xf) ** 2, axis=1)


def train(data_root, dataset, seed=0):
    X, y = pooled_train(data_root, dataset)
    idx = np.concatenate([np.where(y == 1)[0],
                          np.random.default_rng(seed).choice(np.where(y == 0)[0],
                          size=min(60000, int(np.sum(y == 0))), replace=False)])
    return AutoencoderBaseline(**CONFIG[dataset], seed=seed).fit(X[idx], y[idx])


if __name__ == "__main__":
    cli("AE", train)
