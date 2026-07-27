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
5. optionally run baselines,
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

## Numerical Baselines (Table 5)

The numerical SMART baselines (RF, NN, EC, AE, LSTM, MVTRF, MSFRD) are packaged
under `baselines_repro/`, one file per method, each faithful to its source paper
and using the **same implementation and hyper-parameters for MB1 and MB2**.

Because the processed windows are large, the baselines are trained and evaluated
on sampled splits (all failed windows kept, healthy downsampled; see
`baselines_repro/sample_splits.py`). Each baseline is trained on the pooled
train + val windows and scored on a fixed imbalanced 1:23 sample; the per-method
evaluation samples and checkpoints are shipped for a one-command rebuild.

```bash
cd baselines_repro
python reproduce_table5.py            # rebuild the Table 5 block from the shipped checkpoints + eval sets
python train_baselines.py --data-root data/splits_sampled --out checkpoints_retrained   # retrain
```

See `baselines_repro/README.md` for details.

## Approximate Runtime Guidance

These are broad artifact-level expectations rather than hard guarantees:

- quick smoke test: minutes
- cached reproduction: minutes
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
