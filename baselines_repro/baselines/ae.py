"""Autoencoder baseline (Chakraborttii & Litz, SoCC'20).

A 1-class dense autoencoder trained on healthy windows only; the per-window
reconstruction error is the anomaly score. Each SMART window is summarised by
per-attribute statistics before encoding, and the bottleneck (40-10-40, tanh)
forces the network to learn the healthy manifold so failing windows reconstruct
poorly. Evaluated on the fixed 1:23 test set pooled across rounds.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import pooled_train, per_attribute_stats, MinMaxScaler, cli  # noqa: E402

CONFIG = {"MB1": {}, "MB2": {}}


class AutoencoderBaseline:
    def __init__(self, hidden=(40, 10, 40), epochs=300, seed=0):
        self.hidden = hidden; self.epochs = epochs; self.seed = seed

    def _feats(self, X):
        return per_attribute_stats(X)

    def fit(self, X, y):
        self.scaler = MinMaxScaler().fit(self._feats(X))
        healthy = self.scaler.transform(self._feats(X)[np.asarray(y) == 0])
        self.ae = MLPRegressor(hidden_layer_sizes=self.hidden, activation="tanh", solver="adam",
                               max_iter=self.epochs, n_iter_no_change=15, random_state=self.seed)
        self.ae.fit(healthy, healthy)
        # standardise each feature's reconstruction error by its healthy spread, so
        # attributes the network normally reconstructs tightly dominate the anomaly score
        err = (self.ae.predict(healthy) - healthy) ** 2
        self.err_mu = err.mean(0); self.err_sd = err.std(0) + 1e-9
        return self

    def score(self, X):
        Xf = self.scaler.transform(self._feats(X))
        err = (self.ae.predict(Xf) - Xf) ** 2
        return np.mean((err - self.err_mu) / self.err_sd, axis=1)


def train(data_root, dataset, seed=0):
    X, y = pooled_train(data_root, dataset)
    idx = np.concatenate([np.where(y == 1)[0],
                          np.random.default_rng(seed).choice(np.where(y == 0)[0],
                          size=min(120000, int(np.sum(y == 0))), replace=False)])
    return AutoencoderBaseline(**CONFIG[dataset], seed=seed).fit(X[idx], y[idx])


if __name__ == "__main__":
    cli("AE", train)
