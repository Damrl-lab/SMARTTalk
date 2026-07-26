"""Neural-network (MLP) baseline (Alter et al., SC'19).

Per-attribute statistical features, majority class downsampled to 1:1, and a
feed-forward classifier. Evaluated on the fixed 1:23 test set pooled over rounds.
"""
from __future__ import annotations

from pathlib import Path
import sys

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import pooled_train, downsample, per_attribute_stats, cli  # noqa: E402

CONFIG = {
    "MB1": {"hidden_layer_sizes": (128, 64), "max_iter": 200},
    "MB2": {"hidden_layer_sizes": (128, 64), "max_iter": 200},
}


class NeuralNetBaseline:
    def __init__(self, hidden_layer_sizes=(128, 64), max_iter=200, seed=0):
        self.hidden_layer_sizes = hidden_layer_sizes; self.max_iter = max_iter; self.seed = seed

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(per_attribute_stats(X))
        self.clf = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, activation="relu",
                                 solver="adam", max_iter=self.max_iter, early_stopping=True,
                                 n_iter_no_change=10, random_state=self.seed)
        self.clf.fit(self.scaler.transform(per_attribute_stats(X)), y)
        return self

    def score(self, X):
        return self.clf.predict_proba(self.scaler.transform(per_attribute_stats(X)))[:, 1]


def train(data_root, dataset, seed=0):
    X, y = pooled_train(data_root, dataset)
    idx = downsample(y, neg_per_pos=1.0, seed=seed)
    return NeuralNetBaseline(**CONFIG[dataset], seed=seed).fit(X[idx], y[idx])


if __name__ == "__main__":
    cli("NN", train)
