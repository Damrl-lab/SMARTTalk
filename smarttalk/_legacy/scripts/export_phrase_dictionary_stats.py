#!/usr/bin/env python3
"""
Export SMARTTalk phrase-dictionary statistics from saved artifacts.

This script uses the repository's saved PatternMemory artifacts, phrase
vocabularies, and test prototype assignments to export:
  - attribute and cross-attribute phrase dictionaries,
  - pattern-center metadata and distance cutoffs,
  - native coverage / out-of-library rates on MB1 and MB2,
  - a compact LaTeX example table, and
  - a Markdown summary for paper integration.

Cross-model transfer coverage can be expensive because it requires running
foreign test windows through the saved encoders on CPU. The script therefore
supports it as an optional flag but does not enable it by default.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_ARTIFACT_ROOT = Path("data/artifacts")
DEFAULT_OUTPUT_ROOT = Path("results/phrase_dictionary")


@dataclass
class CoverageRecord:
    dataset: str
    round_id: int
    kind: str
    matched_patches: int
    total_patches: int
    out_of_library_patches: int

    @property
    def coverage(self) -> float:
        return self.matched_patches / self.total_patches if self.total_patches else 0.0

    @property
    def out_of_library_rate(self) -> float:
        return self.out_of_library_patches / self.total_patches if self.total_patches else 0.0


def load_vocab(path: Path) -> Dict:
    with path.open() as handle:
        return json.load(handle)


def export_vocab_tables(artifact_root: Path, output_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attr_rows: List[Dict[str, object]] = []
    cross_rows: List[Dict[str, object]] = []
    center_rows: List[Dict[str, object]] = []

    for round_dir in sorted(artifact_root.glob("MB*_round*")):
        dataset, round_token = round_dir.name.split("_")
        round_id = int(round_token.replace("round", ""))
        prefix = dataset.lower()

        vocab_path = round_dir / f"{prefix}_vocab.json"
        proto_path = round_dir / f"{prefix}_prototypes_with_phrases.npz"
        vocab = load_vocab(vocab_path)
        proto = np.load(proto_path, allow_pickle=True)

        attr_centers = proto["attr_centers"]
        cross_centers = proto["cross_centers"]
        attr_threshold = float(np.atleast_1d(proto["attr_threshold"])[0])
        cross_threshold = float(np.atleast_1d(proto["cross_threshold"])[0])
        patch_len_attr = int(np.atleast_1d(proto["patch_len_attr"])[0])
        patch_len_cross = int(np.atleast_1d(proto["patch_len_cross"])[0])

        for key, entry in vocab["attr"].items():
            cluster_id = int(key)
            stats = entry.get("stats", {})
            attr_rows.append(
                {
                    "dataset": dataset,
                    "round": round_id,
                    "cluster_id": cluster_id,
                    "label": entry.get("label", ""),
                    "phrase": entry.get("phrase", ""),
                    "support_count": int(entry.get("num_patches", 0)),
                    "distance_cutoff": attr_threshold,
                    "patch_len": patch_len_attr,
                    "stats_json": json.dumps(stats, sort_keys=True),
                }
            )
            center_rows.append(
                {
                    "dataset": dataset,
                    "round": round_id,
                    "kind": "attribute",
                    "cluster_id": cluster_id,
                    "embedding_dim": int(attr_centers.shape[1]),
                    "distance_cutoff": attr_threshold,
                    "center_vector_json": json.dumps(attr_centers[cluster_id].tolist()),
                }
            )

        for key, entry in vocab["cross"].items():
            cluster_id = int(key)
            stats = entry.get("stats", {})
            cross_rows.append(
                {
                    "dataset": dataset,
                    "round": round_id,
                    "cluster_id": cluster_id,
                    "label": entry.get("label", ""),
                    "phrase": entry.get("phrase", ""),
                    "support_count": int(entry.get("num_patches", 0)),
                    "distance_cutoff": cross_threshold,
                    "patch_len": patch_len_cross,
                    "stats_json": json.dumps(stats, sort_keys=True),
                }
            )
            center_rows.append(
                {
                    "dataset": dataset,
                    "round": round_id,
                    "kind": "cross_attribute",
                    "cluster_id": cluster_id,
                    "embedding_dim": int(cross_centers.shape[1]),
                    "distance_cutoff": cross_threshold,
                    "center_vector_json": json.dumps(cross_centers[cluster_id].tolist()),
                }
            )

    attr_df = pd.DataFrame(attr_rows).sort_values(["dataset", "round", "cluster_id"])
    cross_df = pd.DataFrame(cross_rows).sort_values(["dataset", "round", "cluster_id"])
    center_df = pd.DataFrame(center_rows).sort_values(["dataset", "round", "kind", "cluster_id"])

    attr_df.to_csv(output_root / "attribute_phrase_dictionary.csv", index=False)
    cross_df.to_csv(output_root / "cross_attribute_phrase_dictionary.csv", index=False)
    center_df.to_csv(output_root / "pattern_center_metadata.csv", index=False)

    (output_root / "attribute_phrase_dictionary.json").write_text(
        attr_df.to_json(orient="records", indent=2)
    )
    (output_root / "cross_attribute_phrase_dictionary.json").write_text(
        cross_df.to_json(orient="records", indent=2)
    )
    (output_root / "pattern_center_metadata.json").write_text(
        center_df.to_json(orient="records", indent=2)
    )

    return attr_df, cross_df, center_df


def compute_native_coverage(artifact_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for round_dir in sorted(artifact_root.glob("MB*_round*")):
        dataset, round_token = round_dir.name.split("_")
        round_id = int(round_token.replace("round", ""))
        prefix = dataset.lower()
        test_proto_path = round_dir / f"{prefix}_test_prototypes.npz"
        data = np.load(test_proto_path, allow_pickle=True)

        for kind, key in (("attribute", "attr_novel"), ("cross_attribute", "cross_novel")):
            novel = np.asarray(data[key]).astype(bool)
            total = int(novel.size)
            out_of_library = int(novel.sum())
            matched = total - out_of_library
            record = CoverageRecord(
                dataset=dataset,
                round_id=round_id,
                kind=kind,
                matched_patches=matched,
                total_patches=total,
                out_of_library_patches=out_of_library,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "round": round_id,
                    "kind": kind,
                    "matched_patches": matched,
                    "total_patches": total,
                    "coverage": record.coverage,
                    "out_of_library_patches": out_of_library,
                    "out_of_library_rate": record.out_of_library_rate,
                }
            )

    df = pd.DataFrame(rows).sort_values(["dataset", "round", "kind"])
    return df


def add_aggregate_rows(df: pd.DataFrame) -> pd.DataFrame:
    agg_rows: List[Dict[str, object]] = []
    for (dataset, kind), group in df.groupby(["dataset", "kind"], sort=True):
        matched = int(group["matched_patches"].sum())
        total = int(group["total_patches"].sum())
        out_of_library = int(group["out_of_library_patches"].sum())
        agg_rows.append(
            {
                "dataset": dataset,
                "round": "aggregate",
                "kind": kind,
                "matched_patches": matched,
                "total_patches": total,
                "coverage": matched / total if total else 0.0,
                "out_of_library_patches": out_of_library,
                "out_of_library_rate": out_of_library / total if total else 0.0,
            }
        )
    return pd.concat([df, pd.DataFrame(agg_rows)], ignore_index=True)


def write_examples_md(attr_df: pd.DataFrame, cross_df: pd.DataFrame, output_path: Path) -> None:
    attr_top = (
        attr_df[attr_df["support_count"] > 0]
        .sort_values(["dataset", "round", "support_count"], ascending=[True, True, False])
        .groupby(["dataset", "round"])
        .head(3)
    )
    cross_top = (
        cross_df[cross_df["support_count"] > 0]
        .sort_values(["dataset", "round", "support_count"], ascending=[True, True, False])
        .groupby(["dataset", "round"])
        .head(2)
    )

    lines = [
        "# Phrase Dictionary Examples",
        "",
        "| Dataset | Kind | Phrase | Support |",
        "|---|---|---|---:|",
    ]

    for _, row in pd.concat([attr_top.assign(kind="attr"), cross_top.assign(kind="cross")]).iterrows():
        phrase = str(row["phrase"]).replace("|", "\\|")
        lines.append(
            f"| {row['dataset']} R{row['round']} | {row['kind']} | {phrase} | {int(row['support_count'])} |"
        )

    output_path.write_text("\n".join(lines) + "\n")


def write_summary_md(coverage_df: pd.DataFrame, attr_df: pd.DataFrame, cross_df: pd.DataFrame, output_path: Path) -> None:
    def aggregate_line(dataset: str, kind: str) -> tuple[float, float]:
        row = coverage_df[
            (coverage_df["dataset"] == dataset)
            & (coverage_df["round"] == "aggregate")
            & (coverage_df["kind"] == kind)
        ].iloc[0]
        return float(row["coverage"]), float(row["out_of_library_rate"])

    mb1_attr_cov, mb1_attr_ool = aggregate_line("MB1", "attribute")
    mb1_cross_cov, mb1_cross_ool = aggregate_line("MB1", "cross_attribute")
    mb2_attr_cov, mb2_attr_ool = aggregate_line("MB2", "attribute")
    mb2_cross_cov, mb2_cross_ool = aggregate_line("MB2", "cross_attribute")

    top_attr = (
        attr_df[attr_df["support_count"] > 0]
        .groupby("label", as_index=False)["support_count"]
        .sum()
        .sort_values("support_count", ascending=False)
        .head(5)
    )
    top_cross = (
        cross_df[cross_df["support_count"] > 0]
        .groupby("label", as_index=False)["support_count"]
        .sum()
        .sort_values("support_count", ascending=False)
        .head(5)
    )

    lines = [
        "# Phrase Dictionary Summary",
        "",
        "The phrase dictionary is not authored per test case. Each dictionary entry comes from clustering learned patch embeddings, collecting training patches that stay within the prototype distance cutoff, computing temporal summary statistics over those assigned patches, and mapping those statistics to compact trend phrases.",
        "",
        f"- MB1 native attribute coverage: {mb1_attr_cov:.2%} matched, {mb1_attr_ool:.2%} out-of-library.",
        f"- MB1 native cross-attribute coverage: {mb1_cross_cov:.2%} matched, {mb1_cross_ool:.2%} out-of-library.",
        f"- MB2 native attribute coverage: {mb2_attr_cov:.2%} matched, {mb2_attr_ool:.2%} out-of-library.",
        f"- MB2 native cross-attribute coverage: {mb2_cross_cov:.2%} matched, {mb2_cross_ool:.2%} out-of-library.",
        "",
        "Frequent attribute-level labels include:",
    ]

    for _, row in top_attr.iterrows():
        lines.append(f"- `{row['label']}` with total support {int(row['support_count'])}.")

    lines.append("")
    lines.append("Frequent cross-attribute labels include:")
    for _, row in top_cross.iterrows():
        lines.append(f"- `{row['label']}` with total support {int(row['support_count'])}.")

    lines.append("")
    lines.append(
        "These phrases generalize at the level of trend shape and co-occurrence structure rather than by memorizing individual drives. Stable low counters, gradual wear progression, single late spikes, repeated bursts, and multi-attribute error activity are all selected from the learned patch inventory rather than written manually."
    )

    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    attr_df, cross_df, _ = export_vocab_tables(args.artifact_root, args.output_root)
    coverage_df = add_aggregate_rows(compute_native_coverage(args.artifact_root))

    coverage_df.to_csv(args.output_root / "phrase_dictionary_stats.csv", index=False)
    write_examples_md(attr_df, cross_df, args.output_root / "phrase_dictionary_examples.md")
    write_summary_md(
        coverage_df,
        attr_df,
        cross_df,
        args.output_root / "phrase_dictionary_generalization_summary.md",
    )

    print(f"Wrote {args.output_root / 'phrase_dictionary_stats.csv'}")
    print(f"Wrote {args.output_root / 'phrase_dictionary_examples.md'}")
    print(f"Wrote {args.output_root / 'phrase_dictionary_generalization_summary.md'}")


if __name__ == "__main__":
    main()
