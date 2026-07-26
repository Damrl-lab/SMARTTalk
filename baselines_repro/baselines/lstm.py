"""LSTM baseline (Hao et al., WAIN'21).

An LSTM sequence classifier over the raw SMART window (PyTorch). When PyTorch is
unavailable, a feed-forward classifier over per-attribute temporal features is
used instead. Evaluated on the fixed 1:23 test set pooled across rounds.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import pooled_train, downsample, per_attribute_stats, cli  # noqa: E402

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

CONFIG = {"MB1": {}, "MB2": {}}


class LSTMBaseline:
    def __init__(self, hidden=64, epochs=20, batch=256, lr=1e-3, seed=0):
        self.hidden = hidden; self.epochs = epochs; self.batch = batch; self.lr = lr; self.seed = seed

    def fit(self, X, y):
        if _HAS_TORCH:
            return self._fit_torch(X, y)
        self.scaler = StandardScaler().fit(per_attribute_stats(X))
        self.clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=80, early_stopping=True,
                                 random_state=self.seed)
        self.clf.fit(self.scaler.transform(per_attribute_stats(X)), y)
        return self

    def score(self, X):
        if _HAS_TORCH:
            return self._score_torch(X)
        return self.clf.predict_proba(self.scaler.transform(per_attribute_stats(X)))[:, 1]

    def _fit_torch(self, X, y):
        torch.manual_seed(self.seed)
        self.mu = np.nan_to_num(X.reshape(-1, X.shape[2]).mean(0))
        self.sd = np.nan_to_num(X.reshape(-1, X.shape[2]).std(0)); self.sd[self.sd < 1e-6] = 1.0
        Xn = np.nan_to_num((X - self.mu) / self.sd).astype(np.float32)
        F = X.shape[2]

        class Net(nn.Module):
            def __init__(s):
                super().__init__(); s.lstm = nn.LSTM(F, self.hidden, batch_first=True)
                s.fc = nn.Linear(self.hidden, 1)
            def forward(s, x):
                o, _ = s.lstm(x); return s.fc(o[:, -1, :]).squeeze(-1)

        self.net = Net()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        pw = torch.tensor([float(np.sum(y == 0)) / max(1.0, float(np.sum(y == 1)))])
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        Xt = torch.from_numpy(Xn); yt = torch.from_numpy(np.asarray(y).astype(np.float32))
        for _ in range(self.epochs):
            perm = torch.randperm(len(Xt)); self.net.train()
            for i in range(0, len(Xt), self.batch):
                b = perm[i:i + self.batch]
                opt.zero_grad(); loss = lossf(self.net(Xt[b]), yt[b]); loss.backward(); opt.step()
        return self

    def _score_torch(self, X):
        Xn = np.nan_to_num((X - self.mu) / self.sd).astype(np.float32)
        self.net.eval()
        with torch.no_grad():
            return torch.sigmoid(self.net(torch.from_numpy(Xn))).numpy()


def train(data_root, dataset, seed=0):
    X, y = pooled_train(data_root, dataset)
    idx = downsample(y, neg_per_pos=1.0, seed=seed)
    return LSTMBaseline(**CONFIG[dataset], seed=seed).fit(X[idx], y[idx])


if __name__ == "__main__":
    cli("LSTM", train)
