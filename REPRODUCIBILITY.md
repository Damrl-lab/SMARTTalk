# REPRODUCIBILITY

## Seeds

Key defaults used in the packaged code:

- sampled-test seed: `2026`
- healthy-to-failed test ratio: `23`
- default ablation window values: `10, 20, 30, 40, 50`
- default ablation patch values: `2, 4, 5, 10, 15`

## Quick Reproduction

```bash
bash scripts/07_reproduce/reproduce_quick.sh
```

Expected behavior:

- uses only bundled sample data and cached outputs,
- validates sample `.npz` schema,
- checks important artifact paths,
- regenerates paper tables from cached state.

## Cached Reproduction

```bash
bash scripts/07_reproduce/reproduce_from_cache.sh
```

Expected behavior:

- uses cached outputs only,
- uses the matching bundled `data/splits/*/test.npz` files when cached
  evaluation against bundled SMARTTalk artifacts is needed,
- rewrites the paper tables and cached figure outputs,
- does not require raw dataset download,
- does not require API keys.

## Full Reproduction

```bash
bash scripts/07_reproduce/reproduce_full.sh
```

Expected behavior:

1. preprocess raw SMART data,
2. build temporal splits,
3. build the fixed sampled test set,
4. run offline pattern-memory construction,
5. optionally reproduce the numerical baselines (see *Numerical Baseline Reproduction* below),
6. run SMARTTalk / Raw-LLM / Heuristic-LLM inference,
7. regenerate the paper tables,
8. run N/L ablations.

For the full workflow, the recommended path is to regenerate the processed
splits and offline artifacts together rather than mixing newly generated
`test.npz` files with older cached prototype-assignment artifacts.

The default MB1 and MB2 configs target round 1. To run other rounds, copy one
of the default configs and update its `round` field.

The paper-level Table 5 and Table 6 numbers are aggregated across rounds 1, 2,
and 3. The single-config convenience wrappers are useful for one dataset / one
round local runs, while the paper tables summarize the aggregate over all three
rounds.

## Numerical Baseline Reproduction

The RF, NN, EC, AE, LSTM, MVTRF, and MSFRD rows of Table 5 are reproduced by the
self-contained `baselines_repro/` package, which bundles the trained checkpoints
and the fixed 1:23 evaluation sets, so no raw dataset or API keys are required.

```bash
cd baselines_repro

# rebuild the baseline block from the bundled checkpoints -> results/table5_reproduced.csv
python reproduce_table5.py

# a single baseline (loads its checkpoint and reports P, R, F0.5, FPR, FNR)
python baselines/rf.py --dataset MB1

# optional: retrain the models from data/splits, then rebuild the table on them
python train_baselines.py --out checkpoints_retrained
python reproduce_table5.py --checkpoints checkpoints_retrained
```

Every baseline is scored on a fixed imbalanced 1:23 (failed:healthy) sampled test
set pooled across rounds 1-3, using the same failed windows and a shared healthy
sample. Reproducing from the bundled checkpoints is deterministic; models
retrained from scratch land close but not identical, since training depends on
the exact data splits and RNG. See `baselines_repro/README.md`.

## Approximate Runtime Guidance

These are broad artifact-level expectations rather than hard guarantees:

- quick smoke test: minutes
- cached reproduction: minutes
- baseline reproduction from bundled checkpoints: minutes
- offline CNN / clustering rebuild: tens of minutes to hours depending on GPU
- live full LLM evaluation: highly dependent on model serving choice and batch size
- full ablation sweep: the most expensive stage

## Nondeterminism

Some stages may vary slightly due to:

- PyTorch GPU kernels,
- KMeans initialization,
- sampling order if configs are changed,
- live LLM nondeterminism if temperature or serving backend differs.

The artifact keeps deterministic seeds where practical and preserves the paper
tables as canonical snapshots in `configs/paper_tables/`.
