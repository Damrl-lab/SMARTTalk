#!/usr/bin/env python3
"""
Evaluate the Raw-LLM baseline on processed SMART windows.

Raw-LLM: directly verbalises raw SMART values over the 30-day window
(no prototype memory, no heuristics beyond simple statistics).

Prereqs:
  pip install openai numpy

Example:
  python code/core/raw_llm_eval.py \
    --dataset-name MB2 \
    --round 1 \
    --evaluate-all \
    --model-name Llama-3.1-8B-Instruct \
    --base-url http://localhost:8000/v1 \
    --api-key EMPTY
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from openai import OpenAI

from sampled_test_utils import DEFAULT_HEALTHY_PER_FAILED, DEFAULT_SAMPLE_SEED, select_eval_indices


# ============================================================
# Strong SYSTEM prompt for Raw-LLM baseline
# ============================================================

RAW_SYSTEM_PROMPT_TEMPLATE = """
You are RawSMART-LLM, an expert assistant that assesses SSD health from
raw daily SMART values over a __WINDOW_DAYS__-day window.

Your primary goal is **high precision**: avoid false alarms. Only label
a drive as "AT_RISK" when there is strong, multi-attribute evidence of
impending failure. It is acceptable for recall to be lower if that helps
reduce false positives.

1. DATA FORMAT AND CONTEXT

You are given raw SMART attribute trajectories for a single SSD over a
fixed __WINDOW_DAYS__-day history. For each SMART attribute r_X, you see:

- A list of __WINDOW_DAYS__ daily values over the window.
- Simple statistics summarising the sequence:
    min, max, mean, last value, and delta = last - first.

Treat the provided sequence as day 0, 1, ..., __LAST_DAY__.

Attributes are labeled with their IDs and (when known) human-readable
names, for example:

  - r_5 (Reallocated sector count)
  - r_187 (Reported uncorrectable errors)
  - r_180 (Available reserved blocks)
  - r_177 (Wear range delta)
  - r_9 (Power on hours)
  - r_12 (Power cycle count)
  - r_241 (Number of blocks written)
  - r_242 (Number of blocks read)

2. TASK DEFINITION

For each __WINDOW_DAYS__-day window, your tasks are:

1) STATUS CLASSIFICATION
   Decide whether the drive is currently:
     - "HEALTHY"  (label 0), or
     - "AT_RISK"  (label 1, likely to fail soon).

   Guidelines:
   - Use "AT_RISK" only when there is clear evidence such as:
       * multiple increasing or spiky error attributes
         (e.g., r_5, r_181, r_182, r_183, r_184, r_187, r_195, r_197, r_199),
       * sharp jumps or persistent growth in error attributes late in
         the __WINDOW_DAYS__-day window,
       * large drops in r_180 (Available reserved blocks),
       * abnormal wear (e.g., large and growing r_177),
       * heavy workload combined with worsening errors.
   - If signals are weak, ambiguous, or only slightly elevated, prefer
     "HEALTHY" to avoid false alarms.

2) TIME-TO-FAILURE (TTF) ESTIMATION
   Only if you judge the drive to be AT_RISK, estimate when it is likely
   to fail in days from "now" (end of the __WINDOW_DAYS__-day window).

   - If status is "HEALTHY", set:
       ttf_days   = null
       ttf_bucket = "NONE"
   - If status is "AT_RISK":
       * Use a rough integer estimate for ttf_days (e.g., 3, 10, 40).
       * Map it into one of the discrete buckets:
           "<7"   : failure likely within 1–6 days
           "7-30" : failure likely within 7–30 days
           ">30"  : risk is elevated but likely >30 days

3) EXPLANATION AND RECOMMENDATIONS
   Provide:
   - A short explanation paragraph that:
       (a) names the most important SMART attributes (use exact IDs
           and names when provided),
       (b) states a concern level as exactly one of:
             "LOW concern", "MEDIUM concern", or "SERIOUS concern",
       (c) justifies why this concern level is appropriate, based on
           the __WINDOW_DAYS__-day raw patterns (e.g., late spikes, trends).
   - 1–3 short operator recommendations, such as:
       - "keep monitoring; no immediate action",
       - "schedule drive replacement within 7 days",
       - "migrate critical workloads away from this drive",
       - "trigger detailed diagnostics for this SSD".

3. OUTPUT FORMAT (STRICT)

Think step by step internally, but in your final answer you MUST output
only a single JSON object with no extra commentary and no markdown.

Use this exact schema:

{
  "status": "HEALTHY" or "AT_RISK",
  "concern_level": "LOW concern" | "MEDIUM concern" | "SERIOUS concern",
  "ttf_days": integer or null,
  "ttf_bucket": "<7" | "7-30" | ">30" | "NONE",
  "explanation": "short paragraph following rules (a)-(c)",
  "recommendations": [
    "short actionable sentence 1",
    "short actionable sentence 2"
  ]
}

Constraints:
- If status is "HEALTHY":
    * concern_level should usually be "LOW concern",
    * ttf_days must be null,
    * ttf_bucket must be "NONE".
- If status is "AT_RISK":
    * ttf_days must be a positive integer,
    * ttf_bucket must not be "NONE".
- Never include anything outside the JSON object.
"""


def build_raw_system_prompt(window_days: int) -> str:
    return (
        RAW_SYSTEM_PROMPT_TEMPLATE
        .replace("__WINDOW_DAYS__", str(window_days))
        .replace("__LAST_DAY__", str(window_days - 1))
    )


# ============================================================
# Utilities: JSON parsing, metrics, data loading
# ============================================================

def extract_json(s: str) -> Dict:
    """
    Extract the first {...} block and parse as JSON.
    Raises ValueError if parsing fails.
    """
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model output:\n{s}")
    snippet = s[start:end + 1]
    return json.loads(snippet)


def status_to_int(status: str) -> int:
    """
    Map free-form status string to 0/1.
    Any variant containing 'RISK' is treated as 1.
    """
    status_norm = status.strip().upper().replace("-", "_")
    if "AT_RISK" in status_norm or "RISK" in status_norm:
        return 1
    return 0


def compute_prf1(y_true: List[int], y_pred: List[int]) -> Tuple[float, float, float]:
    """
    Compute precision / recall / F1 for binary labels.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    tp = int(((y_true_arr == 1) & (y_pred_arr == 1)).sum())
    fp = int(((y_true_arr == 0) & (y_pred_arr == 1)).sum())
    fn = int(((y_true_arr == 1) & (y_pred_arr == 0)).sum())

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def f05_score(precision: float, recall: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = 0.25
    return (1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def load_status_labels_from_base(base_npz: np.lib.npyio.NpzFile) -> np.ndarray:
    """
    Try to load status labels (0=healthy, 1=failed) from processed test.npz.
    """
    candidate_keys = ["y", "labels", "y_fail", "label", "status", "y_status"]
    for k in candidate_keys:
        if k in base_npz.files:
            print(f"Found status labels in test.npz key '{k}'")
            return base_npz[k]
    raise RuntimeError(
        "Could not find status labels in test.npz. "
        f"Please ensure one of {candidate_keys} is present."
    )


def get_raw_window_TF(raw_X: np.ndarray, idx: int) -> np.ndarray:
    """
    Return raw window as [T, F] for the given index, handling (T,F) or (F,T)
    layouts. If shapes don't match expectations, return None.
    """
    w = raw_X[idx]
    if w.ndim != 2:
        print(f"WARNING: raw window {idx} has ndim={w.ndim}, expected 2; skipping.")
        return None

    t0, f0 = w.shape
    # Heuristic: treat the *longer* dimension as time if ambiguous.
    if t0 >= f0:
        return w  # assume shape [T, F]
    else:
        return w.T  # assume shape [F, T] -> transpose to [T, F]


# ============================================================
# Raw-LLM summary builder
# ============================================================

def raw_window_to_summary(
    idx: int, window_TF: np.ndarray, feature_names: np.ndarray | None
) -> str:
    """
    Convert one raw window [T, F] into a human-readable description for the LLM.
    We list raw values and simple statistics per attribute.
    """
    T, F = window_TF.shape
    lines: List[str] = []
    lines.append(f"Window index: {idx}")
    lines.append(f"Days covered: 0..{T} (T={T} days)")
    lines.append("")
    lines.append(f"Raw SMART attribute trajectories over the last {T} days:")

    for f_idx in range(F):
        vals = window_TF[:, f_idx]
        v_min = float(np.min(vals))
        v_max = float(np.max(vals))
        v_mean = float(np.mean(vals))
        v_first = float(vals[0])
        v_last = float(vals[-1])
        v_delta = v_last - v_first

        if feature_names is not None and len(feature_names) == F:
            attr_name = str(feature_names[f_idx])
        else:
            attr_name = f"attr_{f_idx}"

        # Compact numeric formatting
        stats_str = (
            f"stats(min={v_min:.3g}, max={v_max:.3g}, "
            f"mean={v_mean:.3g}, first={v_first:.3g}, "
            f"last={v_last:.3g}, delta={v_delta:.3g})"
        )
        vals_rounded = [round(float(x), 3) for x in vals]
        values_str = "[" + ", ".join(str(x) for x in vals_rounded) + "]"

        lines.append(
            f"- {attr_name}: values={values_str}; {stats_str}"
        )

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="MB2",
                        help="Dataset/model name, e.g. MB1 or MB2.")
    parser.add_argument("--round", type=int, default=1,
                        help="Temporal split round number (default: 1)")
    parser.add_argument("--processed-root", type=str, default="data/processed",
                        help="Root directory for processed split folders.")
    parser.add_argument("--processed-dir", type=str, default=None,
                        help="Optional explicit processed split directory for this run.")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Optional cap on the number of failed windows to evaluate when not using --evaluate-all.")
    parser.add_argument("--evaluate-all", action="store_true",
                        help="Evaluate all windows instead of ratio-based sampled evaluation.")
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED,
                        help="Seed used for healthy-window sampling when --evaluate-all is not set.")
    parser.add_argument("--healthy-per-fail", type=float, default=DEFAULT_HEALTHY_PER_FAILED,
                        help="Number of healthy windows to randomly sample per failed window.")
    parser.add_argument("--sampled-indices-csv", type=str, default=None,
                        help="Optional fixed sampled-test CSV built by scripts/build_sampled_test_set.py.")

    # >>> GPT-4o defaults <<<
    parser.add_argument("--model-name", type=str, default="gpt-4o",
                        help="OpenAI model name (e.g., gpt-4o, gpt-4o-mini)")
    parser.add_argument("--base-url", type=str, default="https://api.openai.com/v1",
                        help="OpenAI API base URL")
    parser.add_argument("--api-key", type=str, default="",
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")

    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0)")
    parser.add_argument("--max-tokens", type=int, default=512,
                    help="Max tokens to generate (default: 512)")
    parser.add_argument("--output-jsonl", type=str, default=None,
                        help="Optional JSONL path for per-window predictions.")
    parser.add_argument("--output-metrics-json", type=str, default=None,
                        help="Optional JSON path for aggregate classification metrics.")

    args = parser.parse_args()

    dataset_name = args.dataset_name.strip().upper()
    if dataset_name not in {"MB1", "MB2"}:
        raise ValueError("--dataset-name must be one of: MB1, MB2")

    round_str = f"{dataset_name}_round{args.round}"
    processed_root = Path(args.processed_dir) if args.processed_dir else Path(args.processed_root) / round_str
    base_test_path = processed_root / "test.npz"

    print(f"Loading raw test windows from {base_test_path} ...")
    base_npz = np.load(base_test_path, allow_pickle=True)
    raw_X = base_npz["X"]
    print(f"Raw X shape: {raw_X.shape}")
    sample_window = get_raw_window_TF(raw_X, 0)
    if sample_window is None:
        raise RuntimeError("Unable to infer test window length from raw_X.")
    window_days = int(sample_window.shape[0])

    feature_names = None
    if "features" in base_npz.files:
        feature_names = base_npz["features"]
        print(f"Loaded feature names with shape: {feature_names.shape}")

    # Load status labels
    y_status = load_status_labels_from_base(base_npz)
    y_status = np.asarray(y_status).astype(int)
    N = raw_X.shape[0]
    assert y_status.shape[0] == N, "Label length mismatch."

    print(f"Total test windows: {N}")
    fail_idx = np.where(y_status == 1)[0]
    healthy_idx = np.where(y_status == 0)[0]
    print(f"Total failed windows : {len(fail_idx)}")
    print(f"Total healthy windows: {len(healthy_idx)}")

    selected, num_fail, num_healthy, sampling_meta = select_eval_indices(
        y_status=y_status,
        dataset_name=dataset_name,
        round_id=args.round,
        evaluate_all=bool(args.evaluate_all),
        num_samples=args.num_samples,
        sample_seed=args.sample_seed,
        healthy_per_fail=args.healthy_per_fail,
        sampled_indices_csv=args.sampled_indices_csv,
    )
    total = len(selected)
    if sampling_meta["selection_mode"] == "all_test_windows":
        print(f"Evaluating all {total} windows ({num_fail} failed, {num_healthy} healthy).")
    elif sampling_meta["selection_mode"] == "fixed_sampled_indices":
        print(
            f"Evaluating on fixed sampled test set with {total} windows "
            f"({num_fail} failed, {num_healthy} healthy) from {sampling_meta['sampled_indices_csv']}."
        )
    else:
        print(f"Evaluating on {total} windows "
              f"({num_fail} failed, {num_healthy} healthy, shuffled; "
              f"healthy_per_fail={args.healthy_per_fail:g}).")

    # Use CLI flag if given, otherwise fall back to env var
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key provided. Pass --api-key or set OPENAI_API_KEY."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,  # default is https://api.openai.com/v1
    )


    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    prediction_records = []

    for rank, idx in enumerate(selected, start=1):
        print("=" * 80)
        print(f"Window {rank}/{total} (global index {idx})")
        gt = int(y_status[idx])
        print(f"Ground truth status: {gt} (0=HEALTHY, 1=FAILED)")

        window_TF = get_raw_window_TF(raw_X, idx)
        if window_TF is None:
            print("Skipping this window due to invalid shape.")
            continue

        summary_text = raw_window_to_summary(idx, window_TF, feature_names)

        user_message = (
            f"You are given raw daily SMART logs for one SSD over a {window_days}-day window. "
            "Use them to decide whether the drive is HEALTHY or AT_RISK, and only "
            "if AT_RISK, estimate time-to-failure as described.\n\n"
            "Raw SMART window:\n"
            + summary_text
        )

        # Build messages; some models (e.g., Gemma) may not support a "system" role.
        model_lower = args.model_name.lower()
        if "gemma" in model_lower:
            # For Gemma, fold the system prompt into the user content
            full_user_content = build_raw_system_prompt(window_days) + "\n\n" + user_message
            messages = [
                {"role": "user", "content": full_user_content},
            ]
        else:
            # Default: use separate system + user messages
            messages = [
                {"role": "system", "content": build_raw_system_prompt(window_days)},
                {"role": "user", "content": user_message},
            ]

        response = client.chat.completions.create(
            model=args.model_name,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )


        raw_output = response.choices[0].message.content
        # print("Raw LLM output:")
        # print(raw_output)

        try:
            obj = extract_json(raw_output)
        except Exception as e:
            raise RuntimeError(
                f"Model output for window {idx} was not valid JSON. "
                "This packaged evaluator does not use free-text fallbacks."
            ) from e

        status_str = str(obj.get("status", "HEALTHY"))
        # we ignore concern_level / ttf / explanation / recommendations
        # for classification metrics here, but you can log them if needed

        pred_status = status_to_int(status_str)
        y_true_all.append(gt)
        y_pred_all.append(pred_status)
        prediction_records.append(
            {
                "index": int(idx),
                "true_status": int(gt),
                "pred_status": int(pred_status),
                "raw_response": obj,
            }
        )

    # --- Overall metrics ---
    n_pred = len(y_pred_all)
    print("\n============================================================")
    print(f"Finished Raw-LLM evaluation.")
    print(f"Requested samples : {total}")
    print(f"Valid predictions : {n_pred}")

    if n_pred > 0:
        y_true_arr = np.asarray(y_true_all, dtype=int)
        y_pred_arr = np.asarray(y_pred_all, dtype=int)

        tp = int(((y_true_arr == 1) & (y_pred_arr == 1)).sum())
        fp = int(((y_true_arr == 0) & (y_pred_arr == 1)).sum())
        fn = int(((y_true_arr == 1) & (y_pred_arr == 0)).sum())
        tn = int(((y_true_arr == 0) & (y_pred_arr == 0)).sum())

        precision, recall, f1 = compute_prf1(y_true_all, y_pred_all)
        f05 = f05_score(precision, recall)
        fpr = safe_div(fp, fp + tn)
        fnr = safe_div(fn, tp + fn)

        print("\n=== Confusion matrix (sampled windows) ===")
        print(f"TP (true positives,  y=1, pred=1): {tp}")
        print(f"FP (false positives, y=0, pred=1): {fp}")
        print(f"FN (false negatives, y=1, pred=0): {fn}")
        print(f"TN (true negatives,  y=0, pred=0): {tn}")

        print("\n=== Overall classification metrics on sampled windows ===")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F0.5     : {f05:.4f}")
        print(f"FPR      : {fpr:.6f}")
        print(f"FNR      : {fnr:.4f}")
        print(f"F1-score : {f1:.4f}")

        if args.output_jsonl:
            output_jsonl = Path(args.output_jsonl)
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with output_jsonl.open("w", encoding="utf-8") as f_out:
                for record in prediction_records:
                    f_out.write(json.dumps(record) + "\n")
            print(f"Wrote per-window predictions to {output_jsonl}")

        if args.output_metrics_json:
            metrics_payload = {
                "dataset_name": dataset_name,
                "round": args.round,
                "method": "Raw-LLM",
                "model_name": args.model_name,
                "n_examples_evaluated": len(y_pred_all),
                "classification": {
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f05": f05,
                    "fpr": fpr,
                    "fnr": fnr,
                    "f1": f1,
                },
                "sampling": {
                    "evaluate_all": bool(args.evaluate_all),
                    "num_failed_selected": int(num_fail),
                    "num_healthy_selected": int(num_healthy),
                    "healthy_per_fail": float(args.healthy_per_fail),
                    "sample_seed": int(args.sample_seed),
                    "sampled_indices_csv": args.sampled_indices_csv,
                    "selection_mode": sampling_meta["selection_mode"],
                },
            }
            output_metrics_json = Path(args.output_metrics_json)
            output_metrics_json.parent.mkdir(parents=True, exist_ok=True)
            output_metrics_json.write_text(json.dumps(metrics_payload, indent=2) + "\n")
            print(f"Wrote aggregate metrics to {output_metrics_json}")
    else:
        print("No valid predictions collected; cannot compute metrics.")


if __name__ == "__main__":
    main()
