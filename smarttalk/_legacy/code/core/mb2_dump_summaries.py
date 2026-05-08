#!/usr/bin/env python3
"""
Dump SMARTTalk prototype summaries for MB1/MB2 test windows without calling any LLM.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from llm_eval import get_raw_window, infer_artifact_prefix, load_status_labels, window_to_summary
from sampled_test_utils import DEFAULT_HEALTHY_PER_FAILED, DEFAULT_SAMPLE_SEED, select_eval_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="MB2",
                        choices=["MB1", "MB2"],
                        help="Dataset/model name.")
    parser.add_argument("--round", type=int, default=1,
                        help="Temporal split round number (default: 1)")
    parser.add_argument("--artifact-root", type=str, default="data/artifacts",
                        help="Root directory for dataset round artifacts.")
    parser.add_argument("--processed-root", type=str, default="data/processed",
                        help="Root directory for processed split folders.")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Optional cap on the number of failed windows to dump.")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional explicit output path.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED,
                        help="Random seed for healthy-window sampling (default: 42).")
    parser.add_argument("--healthy-per-fail", type=float, default=DEFAULT_HEALTHY_PER_FAILED,
                        help="Number of healthy windows to randomly sample per failed window.")
    parser.add_argument("--sampled-indices-csv", type=str, default=None,
                        help="Optional fixed sampled-test CSV built by scripts/build_sampled_test_set.py.")
    args = parser.parse_args()

    dataset_name = args.dataset_name.strip().upper()
    artifact_prefix = infer_artifact_prefix(dataset_name)
    round_str = f"{dataset_name}_round{args.round}"
    artifact_root = Path(args.artifact_root) / round_str
    processed_root = Path(args.processed_root) / round_str

    proto_with_phrases_path = artifact_root / f"{artifact_prefix}_prototypes_with_phrases.npz"
    test_protos_path = artifact_root / f"{artifact_prefix}_test_prototypes.npz"
    base_test_path = processed_root / "test.npz"

    if args.output is None:
        out_name = f"{artifact_prefix}_test_summaries_round{args.round}.txt"
        out_path = artifact_root / out_name
    else:
        out_path = Path(args.output)

    print(f"[dump_summaries] Loading prototype memory (with phrases) from {proto_with_phrases_path} ...")
    proto_npz = np.load(proto_with_phrases_path, allow_pickle=True)
    attr_phrases = proto_npz["attr_phrases"]
    cross_phrases = proto_npz["cross_phrases"]
    feature_names_proto = proto_npz["feature_names"]
    patch_len_attr = int(proto_npz["patch_len_attr"][0])
    patch_len_cross = int(proto_npz["patch_len_cross"][0])

    print(f"[dump_summaries] Loading test prototype assignments from {test_protos_path} ...")
    test_npz = np.load(test_protos_path, allow_pickle=True)
    attr_protos = test_npz["attr_protos"]      # [N, F, P_attr]
    attr_novel = test_npz["attr_novel"]        # [N, F, P_attr]
    cross_protos = test_npz["cross_protos"]    # [N, P_cross]
    cross_novel = test_npz["cross_novel"]      # [N, P_cross]
    feature_names_test = test_npz["feature_names"]

    if feature_names_proto.tolist() != feature_names_test.tolist():
        print("[dump_summaries] WARNING: feature_names differ between prototypes and test_prototypes; "
              "using test_prototypes feature order.")
    feature_names = feature_names_test

    N, F, P_attr = attr_protos.shape
    print(f"[dump_summaries] Test windows: N={N}, F={F}, P_attr={P_attr}")

    # Raw windows for attaching raw_stats/raw_values in the summaries
    if base_test_path.exists():
        print(f"[dump_summaries] Loading raw test windows from {base_test_path} ...")
        base_npz = np.load(base_test_path, allow_pickle=True)
        raw_X = base_npz["X"]
        print(f"[dump_summaries] Raw X shape: {raw_X.shape}")

        if "features" in base_npz.files:
            raw_features = base_npz["features"]
            if not np.array_equal(feature_names, raw_features):
                print("[dump_summaries] WARNING: feature_names differ between test_prototypes and raw X; "
                      "assuming positional alignment.")
    else:
        print(f"[dump_summaries] WARNING: {base_test_path} not found; "
              "raw_stats/raw_values will be disabled in summaries.")
        raw_X = None

    # Status labels for ratio-based sampling (0 = healthy, 1 = failed)
    y_status = load_status_labels(test_npz, base_test_path)
    y_status = np.asarray(y_status).astype(int)
    assert y_status.shape[0] == N, "Label length mismatch."

    selected, num_fail, num_healthy, sampling_meta = select_eval_indices(
        y_status=y_status,
        dataset_name=dataset_name,
        round_id=args.round,
        evaluate_all=False,
        num_samples=args.num_samples,
        sample_seed=args.seed,
        healthy_per_fail=args.healthy_per_fail,
        sampled_indices_csv=args.sampled_indices_csv,
    )
    total = len(selected)

    if sampling_meta["selection_mode"] == "fixed_sampled_indices":
        print(f"[dump_summaries] Dumping summaries for fixed sampled test set with {total} windows "
              f"({num_fail} failed, {num_healthy} healthy) from {sampling_meta['sampled_indices_csv']}.")
    else:
        print(f"[dump_summaries] Dumping summaries for {total} windows "
              f"({num_fail} failed, {num_healthy} healthy, shuffled; "
              f"healthy_per_fail={args.healthy_per_fail:g}).")
    print(f"[dump_summaries] Writing to {out_path} ...")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f_out:
        for rank, idx in enumerate(selected, start=1):
            gt = int(y_status[idx])

            if raw_X is not None:
                raw_window = get_raw_window(raw_X, idx, F, P_attr, patch_len_attr)
            else:
                raw_window = None

            summary_text = window_to_summary(
                idx=idx,
                attr_protos=attr_protos,
                attr_novel=attr_novel,
                cross_protos=cross_protos,
                cross_novel=cross_novel,
                feature_names=feature_names,
                attr_phrases=attr_phrases,
                cross_phrases=cross_phrases,
                patch_len_attr=patch_len_attr,
                patch_len_cross=patch_len_cross,
                raw_window=raw_window,
            )

            f_out.write("=" * 80 + "\n")
            f_out.write(f"Window {rank}/{total} (global index {idx}, y_status={gt})\n\n")
            f_out.write(summary_text)
            f_out.write("\n\n")

    print("[dump_summaries] Done.")


if __name__ == "__main__":
    main()
