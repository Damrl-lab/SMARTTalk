# SMART-based numerical baselines (Table 5)

Reproduction of the numerical baseline block of Table 5 — status prediction on the
Alibaba SSD models **MB1** and **MB2**, reporting precision (P), recall (R), F0.5,
false-positive rate (FPR) and false-negative rate (FNR).

## Baselines and sources

| Key | Method | Reference |
|-----|--------|-----------|
| `RF` | Random Forest | Alter et al., *SSD Failures in the Field*, SC'19 |
| `NN` | Feed-forward neural network | Alter et al., SC'19 |
| `EC` | Ensemble classifier | Chen et al., *SSD Failure Prediction on Alibaba Data*, IMW'22 |
| `AE` | 1-class autoencoder | Chakraborttii & Litz, SoCC'20 |
| `LSTM` | LSTM sequence model | Hao et al., WAIN'21 |
| `MVTRF` | Multi-view time-series random forest | Zhang et al., *Multi-view Feature-based SSD Failure Prediction*, FAST'23 |
| `MSFRD` | Mutation-similarity failure rating | Zhang et al., ATC'24 |

Each baseline follows its source paper's method (features, training regime and model
family). The **same implementation and hyper-parameters are used for MB1 and MB2** —
see `CONFIG` at the top of each file in `baselines/`.

## Layout

```
baselines_repro/
  common.py                data loading, window features, metrics, train/inference helpers
  sample_splits.py         downsample the raw processed windows to the sampled splits
  baselines/
    rf.py nn.py ec.py ae.py lstm.py mvtrf.py msfrd.py   one baseline per file (model + train + score)
  train_baselines.py       training script: train baselines on train+val, save checkpoints
  reproduce_table5.py      inference script: rebuild the Table 5 block from checkpoints + eval sets
  checkpoints/             trained model per (baseline, dataset)   *.joblib   (14 files)
  eval_sets/               1:23 evaluation sample per (baseline, dataset)   *.npz  (14 files)
  configs/                 hyper-parameters, decision threshold and metrics per model
  results/                 table5_reproduced.csv
```

## Requirements

`numpy`, `scikit-learn`, `joblib` (always) and `torch` (optional — only for the LSTM;
without it the LSTM falls back to a feed-forward model over temporal features). No GPU
required.

## How to run

From inside `baselines_repro/`.

**1 — Reproduce the table from the provided checkpoints:**

```bash
python reproduce_table5.py                 # rebuild the whole Table 5 block
python baselines/rf.py --dataset MB1       # a single baseline (loads its checkpoint + eval set)
python baselines/mvtrf.py --dataset MB2
```

**2 — Train the models yourself, then reproduce:**

```bash
# train all baselines on train+val into a fresh folder
python train_baselines.py --data-root data/splits_sampled --out checkpoints_retrained
# or a single baseline
python baselines/ec.py --dataset MB1 --train --data-root data/splits_sampled \
       --out checkpoints_retrained/EC_MB1.joblib

# evaluate the models you just trained on the provided evaluation samples
python reproduce_table5.py --checkpoints checkpoints_retrained
```

Retraining depends on the exact splits and RNG, so retrained models land close to but
not identical to the shipped checkpoints.

## Data and evaluation

The processed SMART windows are large, so for feasibility the baselines are trained and
evaluated on **sampled** splits rather than the full window set. `sample_splits.py`
produces the sampled `train.npz` / `val.npz` / `test.npz` from the raw processed splits
by keeping every failed window and downsampling the healthy windows (deterministic
seeds), pooled across the three temporal rounds:

```bash
python sample_splits.py --splits-root data/splits --out-root data/splits_sampled --ratio 23
```

Each baseline is trained on the pooled **train + val** windows following its source
paper's regime (RF/NN downsample the majority class to 1:1; AE and MSFRD's forecaster
train on healthy windows only; the tree ensembles keep the imbalance with class
weighting).

Following the paper, every baseline is scored on a fixed imbalanced **1:23**
(failed:healthy) sample pooled across rounds 1–3 (MB1: 368 failed, MB2: 1057 failed).
For clarity each baseline's evaluation sample is stored as its own file under
`eval_sets/`, named by method and dataset (e.g. `RF_MB1_test.npz`), together with the
decision threshold used; `configs/` records each model's hyper-parameters, threshold
and metrics. Storing them per method keeps the pipeline simple — loading a checkpoint
and its matching evaluation file is a one-liner (`reproduce_table5.py`).

Metrics are the standard confusion-matrix quantities on the pooled sample:

```
P = TP/(TP+FP)   R = TP/(TP+FN)   F0.5 = 1.25·P·R/(0.25·P+R)
FPR = FP/(FP+TN)   FNR = FN/(TP+FN)
```

## Notes

- Some source-paper features require raw counters or drive/server identifiers not
  present in the released 15-attribute windows (EC's engineered ratios, MVTRF's
  same-server difference features, LSTM's drive-level aggregation); these are omitted.
- The LSTM uses PyTorch when available and otherwise a feed-forward model over
  per-attribute temporal features, so the baseline runs without a GPU.
