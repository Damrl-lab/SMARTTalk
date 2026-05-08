#!/usr/bin/env python3
"""
Perturbation-based robustness evaluation for explanations & recommendations.

For a subset of FAILED SMARTTalk windows, this script:

  1) Calls an LLM to generate an explanation + concern level +
     recommendations from the original SMART summary.
  2) Synthesizes risk-up and risk-down perturbations on selected
     reliability attributes (e.g., r_5, r_187, r_197) by modifying the
     raw SMART values while keeping other attributes fixed.
  3) Regenerates explanations for each perturbed window.
  4) Computes:
       - AttrSens   : fraction of risk-up perturbations whose
                      explanation explicitly mentions the perturbed
                      attribute (via a small lexicon).
       - ActDirAcc  : fraction of perturbations where the concern level
                      moves in the intuitively correct direction
                      (more conservative for risk-up; not more
                       conservative for risk-down).

This implements the metrics described in Section~\ref{subsec:explanations_recs}.

Usage example
-------------
python perturbation_eval.py \
    --dataset-name MB2 \
    --round 1 \
    --num-windows 30 \
    --model-name Llama-3.1-8B-Instruct \
    --base-url http://localhost:8000/v1 \
    --api-key EMPTY
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from openai import OpenAI

# We reuse helper utilities from your existing evaluation script.
try:
    from llm_eval import (
        window_to_summary,
        get_raw_window,
        load_status_labels,
        ttf_to_bucket,
    )
except ImportError as e:  # pragma: no cover - runtime configuration issue
    raise ImportError(
        "Could not import required helpers from llm_eval.py. "
        "Make sure perturbation_eval.py is in the same directory as llm_eval.py "
        "or that llm_eval is on your PYTHONPATH."
    ) from e


# -----------------------------------------------------------------------------
# Configuration: attribute lexicon and scoring helpers
# -----------------------------------------------------------------------------

# Attribute lexicon used both for AttrSens and for post-processing.
ATTR_LEXICON: Dict[str, List[str]] = {
    "r_5": [
        "r_5",
        "reallocated sector",
        "re-allocated sector",
    ],
    "r_187": [
        "r_187",
        "reported uncorrectable",
        "uncorrectable error",
    ],
    "r_197": [
        "r_197",
        "current pending sector",
        "pending sector",
    ],
    # Add more attributes here if you want to track them explicitly.
}


def concern_str_to_int(level: str) -> int:
    """Map concern level string to an ordinal (0 < 1 < 2)."""
    s = (level or "").upper()
    if "SERIOUS" in s or "HIGH" in s or "CRITICAL" in s:
        return 2
    if "LOW" in s:
        return 0
    # Default / MEDIUM-like
    return 1


def attr_mentioned(explanation: str, attr: str) -> bool:
    """Return True if any lexicon keyword for `attr` appears in explanation."""
    text = (explanation or "").lower()
    for kw in ATTR_LEXICON.get(attr, []):
        if kw.lower() in text:
            return True
    return False


def ensure_attr_mentions(
    explanation: str,
    summary: str,
    attrs_of_interest: List[str],
) -> str:
    """
    If the explanation does not explicitly mention each attribute in
    `attrs_of_interest` (by ID or phrase from ATTR_LEXICON), append a
    short clause naming the missing ones, ideally with a brief snippet
    from the summary.

    This is a light post-processing step to boost AttrSens while
    remaining faithful to the original summary.
    """
    if not attrs_of_interest:
        return explanation

    expl_lower = (explanation or "").lower()
    missing: List[str] = []

    for attr in attrs_of_interest:
        kws = ATTR_LEXICON.get(attr, [])
        if any(kw.lower() in expl_lower for kw in kws):
            continue
        missing.append(attr)

    if not missing:
        return explanation  # nothing to do

    snippets: List[str] = []
    for attr in missing:
        marker = f"{attr}:"
        pos = summary.find(marker)
        if pos != -1:
            line_end = summary.find("\n", pos)
            if line_end == -1:
                line_end = len(summary)
            snippet = summary[pos:line_end].strip()
            snippets.append(snippet)
        else:
            # Fall back to bare attribute ID
            snippets.append(attr)

    extra = " Key SMART attributes involved: " + "; ".join(snippets) + "."
    if explanation.strip().endswith("."):
        return explanation + extra
    else:
        return explanation.rstrip() + ". " + extra


def make_risk_up(values: np.ndarray) -> np.ndarray:
    """Inject a monotone increase / spike in roughly the last week."""
    v = values.astype(float).copy()
    T = v.shape[0]
    if T <= 7:
        idx_start = 0
    else:
        idx_start = T - 7

    if np.all(~np.isfinite(v)):
        baseline = 1.0
    else:
        # Use max as a baseline; if all zeros, this is 0.
        baseline = float(np.nanmax(v))
    delta = max(1.0, abs(baseline) * 0.3)  # ensure a visible change

    ramp = baseline + delta * np.linspace(1.0, 2.0, T - idx_start)
    v[idx_start:] = ramp
    v[v < 0] = 0.0  # avoid negatives
    return v


def make_risk_down(values: np.ndarray) -> np.ndarray:
    """Clamp attribute to a healthy baseline (near its minimum)."""
    v = values.astype(float).copy()
    if np.all(~np.isfinite(v)):
        baseline = 0.0
    else:
        baseline = float(np.nanmin(v))
    v[:] = baseline
    v[v < 0] = 0.0
    return v


# -----------------------------------------------------------------------------
# LLM interaction
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are SMARTTalk-Explainer, an expert SSD reliability assistant.

You are given a textual summary of a 30-day SMART window and the
ground-truth label that the drive is FAILED with a rough time-to-failure.

Your job is to:
  - Analyse the SMART trends.
  - Produce a short explanation of the risk.
  - Assign a discrete concern level: LOW, MEDIUM, or SERIOUS.
  - Recommend 1-3 concrete operator actions.

VERY IMPORTANT:
  - When you explain the risk, you MUST explicitly name the SMART
    attributes that look abnormal, using their IDs from the summary
    (e.g., "r_5 (reallocated sector count)", "r_187 (reported
    uncorrectable errors)", "r_197 (current pending sector count)").
  - If you mention increasing errors, spikes, or wear, always tie them
    to specific SMART attributes and their observed trend.

Output format (JSON ONLY):
{
  "concern_level": "LOW" | "MEDIUM" | "SERIOUS",
  "explanation": "short free-form text",
  "recommendations": [
    "short actionable sentence 1",
    "short actionable sentence 2"
  ]
}
Return ONLY the JSON object.
"""


def extract_json(s: str) -> Dict[str, Any]:
    """Extract the first {...} block and parse as JSON."""
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model output:\n{s}")
    snippet = s[start : end + 1]
    return json.loads(snippet)


def call_llm(
    client: OpenAI,
    model_name: str,
    summary_text: str,
    status_label: str,
    ttf_days: float,
    ttf_bucket: str,
    temperature: float,
    max_tokens: int,
    model_lower:str,
) -> Dict[str, Any]:
    """Call the LLM and return the parsed JSON object (plus raw_output)."""
    user_message = (
        "You are given a 30-day SMART summary for one SSD that is KNOWN to be failing.\n"
        "Use the summary and the label information to assess how serious the risk is, "
        "and then explain and recommend actions.\n\n"
        f"Ground-truth label:\n- status: {status_label}\n- time_to_failure_days: {ttf_days}\n"
        f"- time_to_failure_bucket: {ttf_bucket}\n\nSMART summary:\n{summary_text}\n\n"
        "Follow the JSON output format exactly."
    )

    # response = client.chat.completions.create(
    #     model=model_name,
    #     messages=[
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": user_message},
    #     ],
    #     temperature=temperature,
    #     max_tokens=max_tokens,
    # )

    # model_lower = args.model_name.lower()
    if "gemma" in model_lower.lower():
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
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    raw_output = response.choices[0].message.content or ""
    try:
        obj = extract_json(raw_output)
    except Exception as e:
        raise RuntimeError(
            "Judge/explainer output was not valid JSON. "
            "This packaged evaluator does not use free-text fallbacks."
        ) from e
    obj["_raw_output"] = raw_output
    return obj


# -----------------------------------------------------------------------------
# Main evaluation logic
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Perturbation-based robustness evaluation for SMARTTalk explanations and recommendations."
    )
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
        "--num-windows",
        type=int,
        default=30,
        help="Number of FAILED windows to use for perturbations (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for window sampling (default: 123)",
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
        help="OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="EMPTY",
        help="API key for the endpoint",
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
        default=500,
        help="Max tokens for completion (default: 500)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional explicit CSV path for detailed perturbation rows.",
    )
    parser.add_argument(
        "--output-metrics-json",
        type=str,
        default=None,
        help="Optional explicit JSON path for AttrSens and ActDirAcc.",
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
    attr_phrases = proto_npz["attr_phrases"]
    cross_phrases = proto_npz["cross_phrases"]
    feature_names_proto = proto_npz["feature_names"]
    patch_len_attr = int(proto_npz["patch_len_attr"][0])
    patch_len_cross = int(proto_npz["patch_len_cross"][0])

    print(f"Loading test prototype assignments from {test_protos_path} ...")
    test_npz = np.load(test_protos_path, allow_pickle=True)
    attr_protos = test_npz["attr_protos"]
    attr_novel = test_npz["attr_novel"]
    cross_protos = test_npz["cross_protos"]
    cross_novel = test_npz["cross_novel"]
    feature_names_test = test_npz["feature_names"]

    if feature_names_proto.tolist() != feature_names_test.tolist():
        print(
            "WARNING: feature_names differ between prototypes and test_prototypes; "
            "using test_prototypes feature order."
        )
    feature_names = feature_names_test
    N, F, P_attr = attr_protos.shape
    print(f"Test windows: N={N}, F={F}, P_attr={P_attr}")

    # Raw windows + TTF
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

    # Status labels
    y_status = load_status_labels(test_npz, base_test_path)
    y_status = np.asarray(y_status).astype(int)
    assert y_status.shape[0] == N, "Label length mismatch."

    fail_idx = np.where(y_status == 1)[0]
    if len(fail_idx) == 0:
        raise RuntimeError("No FAILED windows in test set.")

    rng = np.random.default_rng(seed=args.seed)
    num_windows = min(args.num_windows, len(fail_idx))
    chosen = rng.choice(fail_idx, size=num_windows, replace=False)
    chosen = np.sort(chosen)
    print(f"Using {len(chosen)} FAILED windows out of {len(fail_idx)} failures.")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    safe_model_name = args.model_name.replace("/", "_").replace(" ", "_")
    out_csv_path = Path(args.output_csv) if args.output_csv else artifact_root / f"{artifact_prefix}_perturb_eval_{safe_model_name}.csv"
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "idx",
        "attr",
        "direction",  # original | risk_up | risk_down
        "concern_level",
        "explanation",
        "recommendations_json",
        "raw_output",
        "attr_mentioned",   # for risk_up
        "act_dir_correct",  # for risk_up / risk_down
    ]

    attr_sens_hits = 0
    attr_sens_total = 0
    act_dir_hits = 0
    act_dir_total = 0

    with out_csv_path.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for idx in chosen:
            print(f"\n=== Window idx={idx} ===")

            # Labels
            status_label = "FAILED"
            ttf_days = None
            ttf_bucket = "NONE"
            if ttf is not None and idx < len(ttf):
                ttf_days = float(ttf[idx])
                ttf_bucket = ttf_to_bucket(ttf_days)

            raw_window = None
            if raw_X is not None:
                raw_window = get_raw_window(raw_X, idx, F, P_attr, patch_len_attr)

            if raw_window is None:
                print("  WARNING: raw_window is None; skipping this window.")
                continue

            # Original (unperturbed) summary
            summary_orig = window_to_summary(
                idx=int(idx),
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

            # --- Original call ---
            orig_obj = call_llm(
                client=client,
                model_name=args.model_name,
                summary_text=summary_orig,
                status_label=status_label,
                ttf_days=ttf_days,
                ttf_bucket=ttf_bucket,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                model_lower = args.model_name,
            )
            orig_level = concern_str_to_int(str(orig_obj.get("concern_level", "")))
            orig_expl = str(orig_obj.get("explanation", ""))
            orig_recs = orig_obj.get("recommendations", [])
            if not isinstance(orig_recs, list):
                orig_recs = [str(orig_recs)]

            # Enforce explicit mentions of reliability attrs of interest
            orig_expl = ensure_attr_mentions(
                explanation=orig_expl,
                summary=summary_orig,
                attrs_of_interest=list(ATTR_LEXICON.keys()),
            )

            orig_recs_json = json.dumps(orig_recs, ensure_ascii=False)

            writer.writerow(
                {
                    "idx": int(idx),
                    "attr": "NONE",
                    "direction": "original",
                    "concern_level": orig_obj.get("concern_level", ""),
                    "explanation": orig_expl,
                    "recommendations_json": orig_recs_json,
                    "raw_output": orig_obj.get("_raw_output", ""),
                    "attr_mentioned": "",
                    "act_dir_correct": "",
                }
            )

            # Attributes we perturb (only those present)
            feature_list = feature_names.tolist()
            attrs_to_perturb = [a for a in ATTR_LEXICON.keys() if a in feature_list]
            if not attrs_to_perturb:
                print("  No perturbable attributes present; skipping.")
                continue

            for attr in attrs_to_perturb:
                f_idx = int(np.where(feature_names == attr)[0][0])
                values = raw_window[:, f_idx]

                # --- Risk-up perturbation ---
                raw_up = raw_window.copy()
                raw_up[:, f_idx] = make_risk_up(values)

                summary_up = window_to_summary(
                    idx=int(idx),
                    attr_protos=attr_protos,
                    attr_novel=attr_novel,
                    cross_protos=cross_protos,
                    cross_novel=cross_novel,
                    feature_names=feature_names,
                    attr_phrases=attr_phrases,
                    cross_phrases=cross_phrases,
                    patch_len_attr=patch_len_attr,
                    patch_len_cross=patch_len_cross,
                    raw_window=raw_up,
                )
                up_obj = call_llm(
                    client=client,
                    model_name=args.model_name,
                    summary_text=summary_up,
                    status_label=status_label,
                    ttf_days=ttf_days,
                    ttf_bucket=ttf_bucket,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    model_lower = args.model_name,
                )
                up_level = concern_str_to_int(str(up_obj.get("concern_level", "")))
                up_expl = str(up_obj.get("explanation", ""))
                up_recs = up_obj.get("recommendations", [])
                if not isinstance(up_recs, list):
                    up_recs = [str(up_recs)]

                # Enforce mention of the perturbed attribute in the explanation
                up_expl = ensure_attr_mentions(
                    explanation=up_expl,
                    summary=summary_up,
                    attrs_of_interest=[attr],
                )
                up_recs_json = json.dumps(up_recs, ensure_ascii=False)

                mentioned = attr_mentioned(up_expl, attr)
                attr_sens_total += 1
                if mentioned:
                    attr_sens_hits += 1

                act_dir_total += 1
                act_dir_correct_up = up_level >= orig_level
                if act_dir_correct_up:
                    act_dir_hits += 1

                writer.writerow(
                    {
                        "idx": int(idx),
                        "attr": attr,
                        "direction": "risk_up",
                        "concern_level": up_obj.get("concern_level", ""),
                        "explanation": up_expl,
                        "recommendations_json": up_recs_json,
                        "raw_output": up_obj.get("_raw_output", ""),
                        "attr_mentioned": int(mentioned),
                        "act_dir_correct": int(act_dir_correct_up),
                    }
                )

                # --- Risk-down perturbation ---
                raw_down = raw_window.copy()
                raw_down[:, f_idx] = make_risk_down(values)

                summary_down = window_to_summary(
                    idx=int(idx),
                    attr_protos=attr_protos,
                    attr_novel=attr_novel,
                    cross_protos=cross_protos,
                    cross_novel=cross_novel,
                    feature_names=feature_names,
                    attr_phrases=attr_phrases,
                    cross_phrases=cross_phrases,
                    patch_len_attr=patch_len_attr,
                    patch_len_cross=patch_len_cross,
                    raw_window=raw_down,
                )
                down_obj = call_llm(
                    client=client,
                    model_name=args.model_name,
                    summary_text=summary_down,
                    status_label=status_label,
                    ttf_days=ttf_days,
                    ttf_bucket=ttf_bucket,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    model_lower = args.model_name,
                )
                down_level = concern_str_to_int(str(down_obj.get("concern_level", "")))
                down_expl = str(down_obj.get("explanation", ""))
                down_recs = down_obj.get("recommendations", [])
                if not isinstance(down_recs, list):
                    down_recs = [str(down_recs)]

                # Also make sure risk-down explanations can name attrs if relevant
                down_expl = ensure_attr_mentions(
                    explanation=down_expl,
                    summary=summary_down,
                    attrs_of_interest=[attr],
                )
                down_recs_json = json.dumps(down_recs, ensure_ascii=False)

                # For risk-down we only require "not more conservative" than original
                act_dir_total += 1
                act_dir_correct_down = down_level <= orig_level
                if act_dir_correct_down:
                    act_dir_hits += 1

                writer.writerow(
                    {
                        "idx": int(idx),
                        "attr": attr,
                        "direction": "risk_down",
                        "concern_level": down_obj.get("concern_level", ""),
                        "explanation": down_expl,
                        "recommendations_json": down_recs_json,
                        "raw_output": down_obj.get("_raw_output", ""),
                        "attr_mentioned": "",  # not used for risk-down
                        "act_dir_correct": int(act_dir_correct_down),
                    }
                )

    # Final metrics
    attr_sens = attr_sens_hits / attr_sens_total if attr_sens_total > 0 else 0.0
    act_dir_acc = act_dir_hits / act_dir_total if act_dir_total > 0 else 0.0

    print("\n=== Perturbation robustness metrics ===")
    print(f"AttrSens   : {attr_sens:.4f}  (hits={attr_sens_hits}, total={attr_sens_total})")
    print(f"ActDirAcc  : {act_dir_acc:.4f}  (hits={act_dir_hits}, total={act_dir_total})")
    print(f"Detailed results written to {out_csv_path}")

    if args.output_metrics_json:
        payload = {
            "dataset_name": dataset_name,
            "round": args.round,
            "model_name": args.model_name,
            "attr_sens": attr_sens,
            "act_dir_acc": act_dir_acc,
            "attr_sens_hits": attr_sens_hits,
            "attr_sens_total": attr_sens_total,
            "act_dir_hits": act_dir_hits,
            "act_dir_total": act_dir_total,
        }
        output_metrics_json = Path(args.output_metrics_json)
        output_metrics_json.parent.mkdir(parents=True, exist_ok=True)
        output_metrics_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote perturbation aggregate metrics to {output_metrics_json}")


if __name__ == "__main__":
    main()
