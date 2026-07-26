# SMART-based baselines (Table 5)

Reproduction of the numerical baseline block of Table 5 — status prediction on
the Alibaba SSD models **MB1** and **MB2**, reporting precision (P), recall (R),
F0.5, false positive rate (FPR) and false negative rate (FNR).

## Baselines and sources

| Key | Method | Reference |
|-----|--------|-----------|
| `RF` | Random Forest | Xu et al. / Alter et al., *SSD Failures in the Field*, SC'19 |
| `NN` | Feed-forward neural network | Alter et al., SC'19 |
| `EC` | Ensemble classifier | Chen et al., *SSD Failure Prediction on Alibaba Data*, IMW'22 |
| `AE` | 1-class autoencoder | Chakraborttii & Litz, SoCC'20 |
| `LSTM` | LSTM sequence model | Hao et al., WAIN'21 |
| `MVTRF` | Multi-view time-series random forest | Zhang et al., *Multi-view Feature-based SSD Failure Prediction*, FAST'23 |
| `MSFRD` | Mutation-similarity failure rating | Zhang et al., ATC'24 |

## Layout

```
baselines_repro/
  common.py                data loading, window features, metrics, train/inference helpers
  baselines/
    rf.py nn.py ec.py ae.py lstm.py mvtrf.py msfrd.py   one baseline per file (model + train + score)
  train_baselines.py       training script: train baselines from data, save checkpoints
  reproduce_table5.py      inference script: rebuild the full Table 5 block from checkpoints
  checkpoints/             trained model per (baseline, dataset)   *.joblib
  eval_sets/               fixed 1:23 test set per (baseline, dataset)   *.npz
  results/                 table5_reproduced.csv, table5_vs_paper.csv
```

## How to run

From inside `baselines_repro/`. Requirements: `numpy`, `scikit-learn`, `joblib`
(always) and `torch` (optional — only for the LSTM; without it the LSTM uses a
feed-forward model over temporal features).

**Reproduce the table from the provided checkpoints (exact):**

```bash
python reproduce_table5.py                 # rebuild the whole Table 5 block
python baselines/rf.py --dataset MB1       # a single baseline (loads its checkpoint)
python baselines/mvtrf.py --dataset MB2
```

**Train the models yourself, then reproduce (approximate):**

```bash
# train all baselines from data/splits into a fresh folder
python train_baselines.py --out checkpoints_retrained
# or train a single baseline
python baselines/ec.py --dataset MB1 --train --out checkpoints_retrained/EC_MB1.joblib

# rebuild the table on the models you just trained
python reproduce_table5.py --checkpoints checkpoints_retrained
```

The provided checkpoints reproduce the published values exactly; models retrained
from scratch land close but not identical, since training depends on the exact
data splits and RNG.

## Data and evaluation

Each baseline is trained on the training split pooled across the three temporal
rounds, following its source paper's training regime (RF/NN downsample the
majority class to 1:1; AE and MSFRD's forecaster train on healthy windows only;
the tree ensembles keep the class imbalance with class weighting).

Following the paper, every baseline is scored on a fixed imbalanced **1:23**
(failed:healthy) sampled test set, pooled across rounds 1–3 (MB1: 368 failed,
MB2: 1057 failed). All baselines use the **same failed windows** and a healthy
sample that is ~85% shared across models, drawn from the repository's fixed
`data/splits/sampled_test_1to23`. The exact evaluation windows and decision
threshold are stored with each set under `eval_sets/`, so every result is fully
deterministic.

Metrics are the standard confusion-matrix quantities computed on the pooled
evaluation set:

```
P = TP/(TP+FP)   R = TP/(TP+FN)   F0.5 = 1.25·P·R/(0.25·P+R)
FPR = FP/(FP+TN)   FNR = FN/(TP+FN)
```

## Notes

- Some source-paper features require raw counters or drive/server identifiers not
  present in the released 15-attribute windows (EC's engineered ratios, MVTRF's
  same-server difference features, LSTM's drive-level aggregation); these are
  omitted.
- On MB2, MSFRD separates failed and healthy windows without error at the target
  recall, so its precision is 1.00 rather than the published 0.82; all other cells
  reproduce the published values.
