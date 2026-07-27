"""Mutation-similarity failure-rating baseline (Zhang et al., ATC'24).

A forecaster predicts the tail of each SMART window from its head; the
prediction error (mutation) is classified by similarity to labelled reference
mutations. Evaluated on the fixed 1:23 test set pooled across rounds.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import pooled_train, flatten, MinMaxScaler, cli  # noqa: E402

CONFIG = {
    "MB1": {"k": 35, "ref_cap": 12000, "fc_iter": 60},
    "MB2": {"k": 35, "ref_cap": 12000, "fc_iter": 60},
}


class MSFRDBaseline:
    def __init__(self, t_in=20, t_out=10, k=35, ref_cap=12000, fc_iter=60, seed=0):
        self.t_in = t_in; self.t_out = t_out; self.k = k; self.ref_cap = ref_cap
        self.fc_iter = fc_iter; self.seed = seed

    def _split(self, X):
        return X[:, :self.t_in, :], X[:, self.t_in:self.t_in + self.t_out, :]

    def fit(self, X, y):
        y = np.asarray(y)
        self.scaler = MinMaxScaler().fit(flatten(X))
        Xs = self.scaler.transform(flatten(X)).reshape(X.shape)
        xin, xout = self._split(Xs)
        Ni, No = xin.reshape(len(X), -1), xout.reshape(len(X), -1)
        self.fc = MLPRegressor(hidden_layer_sizes=(256,), activation="relu", solver="adam",
                               max_iter=self.fc_iter, random_state=self.seed)
        self.fc.fit(Ni[y == 0], No[y == 0])
        mut = No - self.fc.predict(Ni)
        self.mscaler = StandardScaler().fit(mut)
        mut_s = self.mscaler.transform(mut)
        rng = np.random.default_rng(self.seed)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        if len(neg) > self.ref_cap:
            neg = rng.choice(neg, size=self.ref_cap, replace=False)
        ref = np.concatenate([pos, neg])
        self.knn = KNeighborsClassifier(n_neighbors=self.k, weights="distance").fit(mut_s[ref], y[ref])
        return self

    def score(self, X):
        Xs = self.scaler.transform(flatten(X)).reshape(X.shape)
        xin, xout = self._split(Xs)
        mut = xout.reshape(len(X), -1) - self.fc.predict(xin.reshape(len(X), -1))
        return self.knn.predict_proba(self.mscaler.transform(mut))[:, 1]


def train(data_root, dataset, seed=0):
    X, y = pooled_train(data_root, dataset)
    idx = np.concatenate([np.where(y == 1)[0],
                          np.random.default_rng(seed).choice(np.where(y == 0)[0],
                          size=min(60000, int(np.sum(y == 0))), replace=False)])
    return MSFRDBaseline(**CONFIG[dataset], seed=seed).fit(X[idx], y[idx])


if __name__ == "__main__":
    cli("MSFRD", train)
