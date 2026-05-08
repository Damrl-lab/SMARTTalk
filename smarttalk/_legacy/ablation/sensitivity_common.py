#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


ROOT = Path(__file__).resolve().parents[3]
FINALIZED_ROOT = ROOT

DEFAULT_WINDOW_DAYS = 30
DEFAULT_PATCH_LEN = 5
DEFAULT_WINDOW_VALUES = [10, 20, 30, 40, 50]
DEFAULT_PATCH_VALUES = [2, 4, 5, 10, 15]
DEFAULT_DATASETS = ["MB1", "MB2"]
DEFAULT_ROUNDS = [1, 2, 3]
DEFAULT_BACKBONES = ["OS3", "PROP"]
DEFAULT_WINDOW_METHODS = ["raw", "heuristic", "smarttalk"]
DEFAULT_PATCH_METHODS = ["smarttalk"]

STUDY_FOLDER = {
    "window": "window_sensitivity",
    "patch": "patch_sensitivity",
}


def unique_sorted_ints(values: Iterable[int]) -> List[int]:
    return sorted({int(value) for value in values})


def with_baseline(values: Iterable[int], baseline: int) -> List[int]:
    merged = list(values)
    merged.append(int(baseline))
    return unique_sorted_ints(merged)


@dataclass(frozen=True)
class SensitivitySetting:
    study: str
    window_days: int
    patch_len: int

    @property
    def study_folder(self) -> str:
        return STUDY_FOLDER[self.study]

    @property
    def slug(self) -> str:
        return f"N{self.window_days}_L{self.patch_len}"

    @property
    def data_root(self) -> Path:
        return ROOT / "data" / self.study_folder / self.slug

    @property
    def processed_root(self) -> Path:
        return self.data_root / "processed"

    @property
    def artifacts_root(self) -> Path:
        return self.data_root / "artifacts"

    @property
    def results_root(self) -> Path:
        return ROOT / "results" / self.study_folder / self.slug

    @property
    def run_root(self) -> Path:
        return self.results_root / "llm_runs" / "table56"


def iter_settings(
    study: str,
    window_values: Iterable[int],
    patch_values: Iterable[int],
    baseline_window: int = DEFAULT_WINDOW_DAYS,
    baseline_patch: int = DEFAULT_PATCH_LEN,
) -> List[SensitivitySetting]:
    if study == "window":
        return [
            SensitivitySetting(study="window", window_days=window_days, patch_len=baseline_patch)
            for window_days in with_baseline(window_values, baseline_window)
        ]
    if study == "patch":
        return [
            SensitivitySetting(study="patch", window_days=baseline_window, patch_len=patch_len)
            for patch_len in with_baseline(patch_values, baseline_patch)
        ]
    raise ValueError(f"Unsupported study: {study}")


def dataset_by_model_root() -> Path:
    return FINALIZED_ROOT / "data" / "raw" / "dataset_by_model"


def failure_tag_path() -> Path:
    return FINALIZED_ROOT / "data" / "raw" / "ssd_failure_tag.csv"
