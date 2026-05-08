# Artifact Manifest

## Overview

This folder stores the non-paper assets required to inspect or reuse SMARTTalk.

## Included Categories

### `checkpoints/`

Contains copied per-round learned encoder and scaler assets from the curated
workspace, organized under `checkpoints/by_round/`.

Common files include:

- `*_attr_autoencoder.pt`
- `*_cross_autoencoder.pt`
- `*_scaler.npz`

### `pattern_memory/`

Holds prototype-memory style assets when promoted or copied into this package.

Key file types:

- `*_prototypes.npz`
- `*_test_prototypes.npz`
- `*_prototypes_with_phrases.npz`

### `phrase_dictionaries/`

Contains vocabulary / phrase outputs such as:

- `*_vocab.json`
- phrase-dictionary CSV / JSON exports

### `cached_predictions/`

Contains cached JSONL / JSON / CSV outputs that support paper-table or
explanation reproduction without rerunning the full pipeline.

### `cached_ablation_results/`

Contains cached ablation figures and supporting summary files.

## Large-File Guidance

- The full raw dataset should stay external and be downloaded separately.
- Full processed split trees are intentionally kept outside this repository.
- Rebuild processed splits locally with the provided preprocessing scripts when
  needed.

## Inventory

A lightweight machine-readable inventory is provided at:

- `artifact_inventory.json`

Use this together with file sizes from the filesystem to inspect what is
included in the repository.
