# Phrase Dictionary Summary

The phrase dictionary is not authored per test case. Each dictionary entry comes from clustering learned patch embeddings, collecting training patches that stay within the prototype distance cutoff, computing temporal summary statistics over those assigned patches, and mapping those statistics to compact trend phrases.

- MB1 native attribute coverage: 93.97% matched, 6.03% out-of-library.
- MB1 native cross-attribute coverage: 89.29% matched, 10.71% out-of-library.
- MB2 native attribute coverage: 95.64% matched, 4.36% out-of-library.
- MB2 native cross-attribute coverage: 99.41% matched, 0.59% out-of-library.

Frequent attribute-level labels include:
- `HIGH_RATE_BURSTS` with total support 3187625.
- `LOW_STABLE` with total support 1191.
- `FAST_RISE` with total support 1147.

Frequent cross-attribute labels include:
- `MIXED_PATTERN` with total support 261339.
- `MULTI_ATTR_TRENDS` with total support 23.

These phrases generalize at the level of trend shape and co-occurrence structure rather than by memorizing individual drives. Stable low counters, gradual wear progression, single late spikes, repeated bursts, and multi-attribute error activity are all selected from the learned patch inventory rather than written manually.
