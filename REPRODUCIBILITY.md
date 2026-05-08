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

- validates sample `.npz` schema,
- checks important artifact paths,
- regenerates paper tables from cached state.

## Cached Reproduction

```bash
bash scripts/07_reproduce/reproduce_from_cache.sh
```

Expected behavior:

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

The default MB1 and MB2 configs target round 1. To run other rounds, copy one
of the default configs and update its `round` field.

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
