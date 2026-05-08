from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


DEFAULT_HEALTHY_PER_FAILED = 23.0
DEFAULT_SAMPLE_SEED = 2026


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def build_sampled_test_indices(
    y_status: Sequence[int],
    *,
    healthy_per_failed: float = DEFAULT_HEALTHY_PER_FAILED,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> np.ndarray:
    y = np.asarray(y_status).astype(int)
    fail_idx = np.where(y == 1)[0]
    healthy_idx = np.where(y == 0)[0]

    if len(fail_idx) == 0 or len(healthy_idx) == 0:
        raise RuntimeError("Need at least one failed and one healthy sample in the test split.")

    rng = np.random.default_rng(seed=seed)
    num_healthy = min(len(healthy_idx), int(round(len(fail_idx) * healthy_per_failed)))
    chosen_healthy = rng.choice(healthy_idx, size=num_healthy, replace=False)

    selected = np.concatenate([fail_idx, chosen_healthy]).astype(int)
    rng.shuffle(selected)
    return selected


def write_sampled_test_tables(
    *,
    processed_root: Path,
    output_dir: Path,
    datasets: Iterable[str] = ("MB1", "MB2"),
    rounds: Iterable[int] = (1, 2, 3),
    healthy_per_failed: float = DEFAULT_HEALTHY_PER_FAILED,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    indices_csv = output_dir / "sampled_test_indices.csv"
    summary_csv = output_dir / "sampling_summary.csv"

    index_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for dataset in datasets:
        for round_id in rounds:
            test_path = processed_root / f"{dataset}_round{round_id}" / "test.npz"
            data = np.load(test_path)
            y = np.asarray(data["y"]).astype(int)

            fail_idx = np.where(y == 1)[0]
            healthy_idx = np.where(y == 0)[0]
            selected = build_sampled_test_indices(
                y,
                healthy_per_failed=healthy_per_failed,
                seed=seed,
            )

            selected_set = set(int(idx) for idx in selected.tolist())
            chosen_fail = int(sum(1 for idx in selected_set if y[idx] == 1))
            chosen_healthy = int(sum(1 for idx in selected_set if y[idx] == 0))

            for rank, idx in enumerate(selected.tolist(), start=1):
                true_status = int(y[idx])
                index_rows.append(
                    {
                        "dataset": dataset,
                        "round": round_id,
                        "split": "test",
                        "test_index": int(idx),
                        "true_status": true_status,
                        "selected_group": "failed" if true_status == 1 else "healthy",
                        "sample_rank": rank,
                        "healthy_per_failed": healthy_per_failed,
                        "seed": seed,
                    }
                )

            summary_rows.append(
                {
                    "dataset": dataset,
                    "round": round_id,
                    "split": "test",
                    "raw_failed": int(len(fail_idx)),
                    "raw_healthy": int(len(healthy_idx)),
                    "selected_failed": chosen_fail,
                    "selected_healthy": chosen_healthy,
                    "selected_total": int(len(selected)),
                    "healthy_per_failed": healthy_per_failed,
                    "selected_healthy_to_failed_ratio": safe_div(chosen_healthy, chosen_fail),
                    "seed": seed,
                }
            )

    with indices_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "round",
                "split",
                "test_index",
                "true_status",
                "selected_group",
                "sample_rank",
                "healthy_per_failed",
                "seed",
            ],
        )
        writer.writeheader()
        writer.writerows(index_rows)

    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "round",
                "split",
                "raw_failed",
                "raw_healthy",
                "selected_failed",
                "selected_healthy",
                "selected_total",
                "healthy_per_failed",
                "selected_healthy_to_failed_ratio",
                "seed",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    return indices_csv, summary_csv


def load_selected_indices(
    sampled_indices_csv: Path,
    *,
    dataset_name: str,
    round_id: int,
) -> np.ndarray:
    rows: List[int] = []
    with sampled_indices_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["dataset"].strip().upper() != dataset_name.strip().upper():
                continue
            if int(row["round"]) != int(round_id):
                continue
            rows.append(int(row["test_index"]))

    if not rows:
        raise RuntimeError(
            f"No sampled test indices found in {sampled_indices_csv} for "
            f"{dataset_name} round {round_id}."
        )
    return np.asarray(rows, dtype=int)


def select_eval_indices(
    *,
    y_status: Sequence[int],
    dataset_name: str,
    round_id: int,
    evaluate_all: bool,
    num_samples: int | None,
    sample_seed: int,
    healthy_per_fail: float,
    sampled_indices_csv: str | None,
) -> tuple[list[int], int, int, Dict[str, object]]:
    y = np.asarray(y_status).astype(int)
    fail_idx = np.where(y == 1)[0]
    healthy_idx = np.where(y == 0)[0]

    if len(fail_idx) == 0 or len(healthy_idx) == 0:
        raise RuntimeError("Need at least one failed and one healthy sample in test set.")

    if evaluate_all:
        selected = np.arange(len(y)).tolist()
        num_fail = int(len(fail_idx))
        num_healthy = int(len(healthy_idx))
        metadata = {
            "selection_mode": "all_test_windows",
            "evaluate_all": True,
            "sampled_indices_csv": None,
            "healthy_per_fail": None,
            "sample_seed": None,
        }
        return selected, num_fail, num_healthy, metadata

    if sampled_indices_csv:
        sampled_path = Path(sampled_indices_csv)
        selected_np = load_selected_indices(
            sampled_path,
            dataset_name=dataset_name,
            round_id=round_id,
        )
        selected = selected_np.tolist()
        num_fail = int(sum(1 for idx in selected if y[idx] == 1))
        num_healthy = int(sum(1 for idx in selected if y[idx] == 0))
        metadata = {
            "selection_mode": "fixed_sampled_indices",
            "evaluate_all": False,
            "sampled_indices_csv": str(sampled_path),
            "healthy_per_fail": float(healthy_per_fail),
            "sample_seed": int(sample_seed),
        }
        return selected, num_fail, num_healthy, metadata

    rng = np.random.default_rng(seed=sample_seed)
    num_fail = len(fail_idx) if num_samples is None else min(len(fail_idx), num_samples)
    if num_fail == len(fail_idx):
        chosen_fail = fail_idx
    else:
        chosen_fail = rng.choice(fail_idx, size=num_fail, replace=False)

    num_healthy = min(len(healthy_idx), int(round(num_fail * healthy_per_fail)))
    chosen_healthy = rng.choice(healthy_idx, size=num_healthy, replace=False)

    selected = np.concatenate([chosen_fail, chosen_healthy])
    rng.shuffle(selected)

    metadata = {
        "selection_mode": "ratio_sampled_runtime",
        "evaluate_all": False,
        "sampled_indices_csv": None,
        "healthy_per_fail": float(healthy_per_fail),
        "sample_seed": int(sample_seed),
    }
    return selected.astype(int).tolist(), int(num_fail), int(num_healthy), metadata
