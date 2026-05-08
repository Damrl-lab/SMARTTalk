# ORGANIZATION_REPORT

## Summary

This package was assembled from the existing SMARTTalk workspace into a cleaner
GitHub-ready artifact repository layout.

## Main Changes

### Added

- top-level artifact docs:
  - `README.md`
  - `ARTIFACT_CLAIMS.md`
  - `DATA_ACCESS.md`
  - `ENVIRONMENT.md`
  - `REPRODUCIBILITY.md`
  - `artifacts/MANIFEST.md`
- a reusable `smarttalk/` package with:
  - config / path helpers,
  - schema checks,
  - status-metric utilities,
  - parser helpers,
  - thin pipeline runners,
  - preserved low-level implementation files under `smarttalk/_legacy/`
- numbered `scripts/01...07` entry points
- bundled sample `.npz` files for smoke tests
- lightweight tests under `tests/`

### Copied Forward

- working low-level code from the earlier curated package
- paper tables and paper figures
- phrase-dictionary outputs
- ablation figures
- checkpoints and cached outputs that were already present locally

### Intentionally Not Bundled in Full GitHub Form

- the full raw Alibaba dataset
- multi-gigabyte processed split trees
- any private API credentials

These are documented in `DATA_ACCESS.md` and `artifacts/MANIFEST.md`.

## Path Normalization

- new wrapper scripts use paths relative to the repository root
- no absolute local machine paths are required at runtime
- external dataset download is explicitly documented instead of assumed

## Scientific Result Policy

- the artifact preserves the finalized paper tables as canonical snapshots
- the packaged Table 5 includes the updated sampled-set FPR/FNR values
- no paper results were intentionally changed during organization

## Known Limitations

- no project-approved open-source license text was present in the original
  workspace, so `LICENSE` is a release placeholder that should be replaced
  before a public GitHub release
- the full raw dataset must be downloaded separately
- live LLM evaluation still depends on API access or local model serving
