#!/usr/bin/env python3
"""
Generate Figure 4 / Figure 5 style prototype visualizations from saved SMARTTalk artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = ROOT / "code" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from learn_vocab_from_prototypes import (
    AttrAutoencoder,
    CrossAutoencoder,
    collect_patches_for_prototypes,
    get_device,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_attr_examples(attr_buckets, attr_vocab, output_root: Path) -> None:
    ranked = sorted(
        [(k, len(v)) for k, v in enumerate(attr_buckets) if len(v) > 0],
        key=lambda item: item[1],
        reverse=True,
    )[:6]
    if not ranked:
        raise RuntimeError("No attribute prototype patches were collected.")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    axes = axes.flatten()
    for ax, (proto_id, _) in zip(axes, ranked):
        stack = np.stack(attr_buckets[proto_id], axis=0)
        for patch in stack[:20]:
            ax.plot(patch, color="#7aa6c2", alpha=0.18, linewidth=1.0)
        ax.plot(stack.mean(axis=0), color="#0d3b66", linewidth=2.0)
        phrase = attr_vocab[str(proto_id)]["phrase"] if str(proto_id) in attr_vocab else attr_vocab[proto_id]["phrase"]
        ax.set_title(phrase, fontsize=10)
        ax.grid(True, alpha=0.25)
    for ax in axes[len(ranked):]:
        ax.set_visible(False)
    fig.suptitle("Representative attribute-level patterns", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_root / "fig_attr_prototypes.png", dpi=300)
    fig.savefig(output_root / "fig_attr_prototypes.pdf")
    plt.close(fig)


def plot_cross_examples(cross_buckets, cross_vocab, feature_names, output_root: Path) -> None:
    ranked = sorted(
        [(k, len(v)) for k, v in enumerate(cross_buckets) if len(v) > 0],
        key=lambda item: item[1],
        reverse=True,
    )[:6]
    if not ranked:
        raise RuntimeError("No cross-attribute prototype patches were collected.")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for ax, (proto_id, _) in zip(axes, ranked):
        stack = np.stack(cross_buckets[proto_id], axis=0)
        mean_patch = stack.mean(axis=0)
        im = ax.imshow(mean_patch, aspect="auto", cmap="viridis")
        phrase = cross_vocab[str(proto_id)]["phrase"] if str(proto_id) in cross_vocab else cross_vocab[proto_id]["phrase"]
        ax.set_title(phrase, fontsize=10)
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names, fontsize=7)
        ax.set_xticks(range(mean_patch.shape[1]))
        ax.set_xticklabels(range(mean_patch.shape[1]), fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[len(ranked):]:
        ax.set_visible(False)
    fig.suptitle("Representative cross-attribute patterns", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_root / "fig_cross_prototypes.png", dpi=300)
    fig.savefig(output_root / "fig_cross_prototypes.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild Figure 4 and Figure 5 prototype plots from saved SMARTTalk artifacts.",
    )
    parser.add_argument("--dataset-name", type=str, default="MB2", choices=["MB1", "MB2"])
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--processed-root", type=str, default="data/processed")
    parser.add_argument("--artifacts-root", type=str, default="data/artifacts")
    parser.add_argument("--output-root", type=str, default="results/paper_figures")
    args = parser.parse_args()

    dataset = args.dataset_name.upper()
    prefix = dataset.lower()
    artifact_root = Path(args.artifacts_root) / f"{dataset}_round{args.round}"
    processed_root = Path(args.processed_root) / f"{dataset}_round{args.round}"
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_npz = np.load(processed_root / "train.npz", allow_pickle=True)
    X_train = train_npz["X"].astype(np.float32)
    train_features = train_npz["features"].tolist()

    scaler = np.load(artifact_root / f"{prefix}_scaler.npz", allow_pickle=True)
    feat_mean = scaler["mean"].astype(np.float32)
    feat_std = scaler["std"].astype(np.float32)
    scaler_features = scaler["features"].tolist()
    if train_features != scaler_features:
        name_to_idx = {name: i for i, name in enumerate(train_features)}
        idxs = [name_to_idx[name] for name in scaler_features]
        X_train = X_train[:, :, idxs]
    X_train_norm = (X_train - feat_mean) / feat_std
    feature_names = scaler_features

    proto = np.load(artifact_root / f"{prefix}_prototypes.npz", allow_pickle=True)
    attr_centers = proto["attr_centers"]
    cross_centers = proto["cross_centers"]
    attr_threshold = float(np.atleast_1d(proto["attr_threshold"])[0])
    cross_threshold = float(np.atleast_1d(proto["cross_threshold"])[0])
    patch_len_attr = int(np.atleast_1d(proto["patch_len_attr"])[0])
    patch_len_cross = int(np.atleast_1d(proto["patch_len_cross"])[0])
    attr_emb_dim = int(np.atleast_1d(proto["attr_emb_dim"])[0])
    cross_emb_dim = int(np.atleast_1d(proto["cross_emb_dim"])[0])

    attr_ae = AttrAutoencoder(patch_len=patch_len_attr, emb_dim=attr_emb_dim)
    cross_ae = CrossAutoencoder(num_features=len(feature_names), patch_len=patch_len_cross, emb_dim=cross_emb_dim)
    attr_ae.load_state_dict(torch.load(artifact_root / f"{prefix}_attr_autoencoder.pt", map_location="cpu"))
    cross_ae.load_state_dict(torch.load(artifact_root / f"{prefix}_cross_autoencoder.pt", map_location="cpu"))

    device = get_device(args.device)
    attr_buckets, cross_buckets = collect_patches_for_prototypes(
        X_raw=X_train,
        X_norm=X_train_norm,
        feature_names=feature_names,
        attr_ae=attr_ae,
        cross_ae=cross_ae,
        attr_centers=attr_centers,
        cross_centers=cross_centers,
        patch_len_attr=patch_len_attr,
        patch_len_cross=patch_len_cross,
        attr_threshold=attr_threshold,
        cross_threshold=cross_threshold,
        device=device,
    )

    vocab = load_json(artifact_root / f"{prefix}_vocab.json")
    plot_attr_examples(attr_buckets, vocab["attribute"], output_root)
    plot_cross_examples(cross_buckets, vocab["cross_attribute"], feature_names, output_root)
    print(f"Wrote Figure 4 / Figure 5 prototype plots to {output_root}")


if __name__ == "__main__":
    main()
