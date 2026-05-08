#!/usr/bin/env python3
"""
Generate explanations and recommendations for failed SMARTTalk windows.

This script reuses the prototype-based summaries from `llm_eval.py`
and asks an LLM to ONLY produce a short natural-language explanation
and a few recommended actions, *given* the ground-truth labels
(status = FAILED and time-to-failure in days + bucket).

Usage example
-------------
python exp_rec_generation.py \
    --dataset-name MB2 \
    --round 1 \
    --num-samples 50 \
    --model-name Llama-3.1-8B-Instruct \
    --base-url http://localhost:8000/v1 \
    --api-key EMPTY

Outputs
-------
A CSV file under:
  data/artifacts/<DATASET>_round{round}/<prefix>_exp_<model>.csv

Each row contains:
  - idx                  : window index in mb2_test_prototypes.npz
  - status_label         : always FAILED (1) for this script
  - ttf_label_days       : ground-truth TTF in days (if available)
  - ttf_label_bucket     : mapped bucket (<7, 7-30, >30, or NONE)
  - summary              : SMARTTalk text summary for this window
  - explanation          : LLM-generated explanation
  - recommendations      : JSON string list of actions
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI

# We reuse helper functions from the existing evaluation script.
# Make sure `llm_eval.py` is in the same directory or on PYTHONPATH.
from llm_eval import (
    window_to_summary,
    get_raw_window,
    load_status_labels,
    ttf_to_bucket,
)


SYSTEM_PROMPT = """
You are SMARTTalk-Explainer, an expert SSD reliability assistant.

You are given:
  (a) a textual summary of a 30-day SMART window for one SSD, and
  (b) the ground-truth label that this drive is FAILED and its
      approximate time-to-failure.

Your job is to:
  - Interpret the SMART trends.
  - Provide a short, technically plausible explanation of *why* the
    drive is failing or at high risk.
  - Provide 1–3 concrete recommended actions for a cloud operator.

VERY IMPORTANT:
  - When you explain *why* the drive is failing, you MUST explicitly
    name the SMART attributes that show problematic behaviour, using
    their IDs from the summary (e.g., "r_5 (reallocated sector count)",
    "r_187 (reported uncorrectable errors)", "r_197 (current pending
    sector count)").
  - Do not talk about “errors” in the abstract; tie your explanation to
    specific SMART attributes and their trends (e.g., spikes, monotone
    increases, bursts).

Rules:
  - Do NOT change the given label or the time-to-failure bucket;
    treat them as correct.
  - Be concise (2–4 sentences of explanation).
  - Recommendations must be practical actions, not generic advice.

Output format (JSON ONLY):
{
  "explanation": "short free-form text",
  "recommendations": [
    "short actionable sentence 1",
    "short actionable sentence 2"
  ]
}
Return ONLY the JSON object, with no extra commentary.
"""



def extract_json(s: str) -> Dict[str, Any]:
    """Extract the first {...} block and parse as JSON."""
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model output:\n{s}")
    snippet = s[start : end + 1]
    return json.loads(snippet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="MB2",
        help="Dataset/model name, e.g. MB1 or MB2.",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=1,
        help="Temporal split round number (default: 1)",
    )
    parser.add_argument(
        "--artifact-root",
        type=str,
        default="data/artifacts",
        help="Root directory for artifacts (default: data/artifacts)",
    )
    parser.add_argument(
        "--processed-root",
        type=str,
        default="data/processed",
        help="Root directory for processed data (default: data/processed)",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help="Optional explicit artifact directory for this run.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Optional explicit processed directory for this run.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of FAILED windows to sample (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Llama-3.1-8B-Instruct",
        help="Model name (OpenAI / vLLM compatible)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="OpenAI-compatible base URL for vLLM",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="EMPTY",
        help="API key for the model endpoint",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=384,
        help="Max tokens for completion (default: 384)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional explicit output CSV path.",
    )

    args = parser.parse_args()

    dataset_name = args.dataset_name.strip().upper()
    if dataset_name not in {"MB1", "MB2"}:
        raise ValueError("--dataset-name must be one of: MB1, MB2")
    artifact_prefix = dataset_name.lower()
    round_str = f"{dataset_name}_round{args.round}"
    artifact_root = Path(args.artifact_dir) if args.artifact_dir else Path(args.artifact_root) / round_str
    processed_root = Path(args.processed_dir) if args.processed_dir else Path(args.processed_root) / round_str

    proto_with_phrases_path = artifact_root / f"{artifact_prefix}_prototypes_with_phrases.npz"
    test_protos_path = artifact_root / f"{artifact_prefix}_test_prototypes.npz"
    base_test_path = processed_root / "test.npz"

    print(f"Loading prototype memory (with phrases) from {proto_with_phrases_path} ...")
    proto_npz = np.load(proto_with_phrases_path, allow_pickle=True)

    attr_phrases = proto_npz["attr_phrases"]       # [K_attr]
    cross_phrases = proto_npz["cross_phrases"]     # [K_cross]
    feature_names_proto = proto_npz["feature_names"]
    patch_len_attr = int(proto_npz["patch_len_attr"][0])
    patch_len_cross = int(proto_npz["patch_len_cross"][0])

    print(f"Loading test prototype assignments from {test_protos_path} ...")
    test_npz = np.load(test_protos_path, allow_pickle=True)

    attr_protos = test_npz["attr_protos"]          # [N, F, P_attr]
    attr_novel = test_npz["attr_novel"]            # [N, F, P_attr]
    cross_protos = test_npz["cross_protos"]        # [N, P_cross]
    cross_novel = test_npz["cross_novel"]          # [N, P_cross]

    feature_names_test = test_npz["feature_names"]
    if feature_names_proto.tolist() != feature_names_test.tolist():
        print(
            "WARNING: feature_names differ between prototypes and test_prototypes; "
            "using test_prototypes feature order."
        )
    feature_names = feature_names_test

    N, F, P_attr = attr_protos.shape
    print(f"Test windows: N={N}, F={F}, P_attr={P_attr}")

    # Load raw windows and TTF labels
    raw_X = None
    ttf = None
    if base_test_path.exists():
        print(f"Loading raw test windows from {base_test_path} ...")
        base_npz = np.load(base_test_path, allow_pickle=True)
        raw_X = base_npz["X"]
        print(f"Raw X shape: {raw_X.shape}")

        if "ttf" in base_npz.files:
            ttf = base_npz["ttf"]
            print(f"Loaded TTF vector with shape: {ttf.shape}")
        else:
            print("WARNING: 'ttf' not found in test.npz; TTF labels will be None.")
    else:
        print(f"WARNING: {base_test_path} not found; raw_X and ttf will be None.")

    # Status labels (0=healthy, 1=failed)
    y_status = load_status_labels(test_npz, base_test_path)
    y_status = np.asarray(y_status).astype(int)
    assert y_status.shape[0] == N, "Label length mismatch."

    # Sample only failed windows
    fail_idx = np.where(y_status == 1)[0]
    if len(fail_idx) == 0:
        raise RuntimeError("No FAILED windows in test set.")

    rng = np.random.default_rng(seed=args.seed)
    num_to_take = min(args.num_samples, len(fail_idx))
    chosen = rng.choice(fail_idx, size=num_to_take, replace=False)
    chosen = np.sort(chosen)

    print(f"Sampling {len(chosen)} FAILED windows out of {len(fail_idx)} failures.")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    safe_model_name = args.model_name.replace("/", "_").replace(" ", "_")
    out_csv_path = Path(args.output_csv) if args.output_csv else artifact_root / f"{artifact_prefix}_exp_{safe_model_name}.csv"
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "idx",
        "status_label",
        "ttf_label_days",
        "ttf_label_bucket",
        "summary",
        "explanation",
        "recommendations_json",
        "raw_output",
    ]

    with out_csv_path.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for i, idx in enumerate(chosen):
            print(f"\n=== Window {i+1}/{len(chosen)} : idx={idx} ===")

            # Ground-truth labels
            status_label = "FAILED"
            ttf_days = None
            ttf_bucket = "NONE"
            if ttf is not None and idx < len(ttf):
                ttf_days = float(ttf[idx])
                ttf_bucket = ttf_to_bucket(ttf_days)

            # Raw window for stats inside summary (if available)
            raw_window = None
            if raw_X is not None:
                raw_window = get_raw_window(raw_X, idx, F, P_attr, patch_len_attr)

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

            user_message = (
                "You are given a 30-day SMART summary for one SSD that is KNOWN to be failing.\\n"
                "Use the summary and the provided label information to explain why it is failing "
                "and what the operator should do.\n\n"
                f"Ground-truth label:\n"
                f"- status: {status_label}\n"
                f"- time_to_failure_days: {ttf_days}\n"
                f"- time_to_failure_bucket: {ttf_bucket}\n\n"
                "SMART summary:\n"
                f"{summary_text}\n\n"
                "Follow the output JSON format exactly."
            )

            # response = client.chat.completions.create(
            #     model=args.model_name,
            #     messages=[
            #         {"role": "system", "content": SYSTEM_PROMPT},
            #         {"role": "user", "content": user_message},
            #     ],
            #     temperature=args.temperature,
            #     max_tokens=args.max_tokens,
            # )

            # Build messages; some models (e.g., Gemma) may not support a "system" role.
            model_lower = args.model_name.lower()
            if "gemma" in model_lower:
                # For Gemma, fold the system prompt into the user content
                full_user_content = SYSTEM_PROMPT + "\n\n" + user_message
                messages = [
                    {"role": "user", "content": full_user_content},
                ]
            else:
                # Default: use separate system + user messages
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ]

            response = client.chat.completions.create(
                model=args.model_name,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            raw_output = response.choices[0].message.content or ""
            # print(raw_output)

            try:
                obj = extract_json(raw_output)
            except Exception as e:
                raise RuntimeError(
                    f"Model output for window {idx} was not valid JSON. "
                    "This packaged evaluator does not use free-text fallbacks."
                ) from e

            explanation = str(obj.get("explanation", ""))
            recs = obj.get("recommendations", [])
            if not isinstance(recs, list):
                recs = [str(recs)]
            recs_json = json.dumps(recs, ensure_ascii=False)

            writer.writerow(
                {
                    "idx": int(idx),
                    "status_label": status_label,
                    "ttf_label_days": ttf_days,
                    "ttf_label_bucket": ttf_bucket,
                    "summary": summary_text,
                    "explanation": explanation,
                    "recommendations_json": recs_json,
                    "raw_output": raw_output,
                }
            )

    print(f"\nWrote {len(chosen)} rows to {out_csv_path}")


if __name__ == "__main__":
    main()
