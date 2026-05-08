#!/usr/bin/env python3
"""
Evaluate SMARTTalk on processed MB1/MB2 test prototypes.

Prereqs:
  pip install openai numpy

Run (example):
  python code/core/llm_eval.py \
      --dataset-name MB2 \
      --round 1 \
      --healthy-per-fail 1 \
      --model-name Llama-3.1-8B-Instruct \
      --base-url http://localhost:8000/v1 \
      --api-key EMPTY


python code/core/llm_eval.py \
  --dataset-name MB2 \
  --round 1 \
  --healthy-per-fail 1 \
  --model-name Phi-4 \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv
import numpy as np
from openai import OpenAI

from sampled_test_utils import DEFAULT_HEALTHY_PER_FAILED, DEFAULT_SAMPLE_SEED, select_eval_indices


# ============================================================
# Strong chain-of-thought style SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT_TEMPLATE = """
You are an assistant that classifies SSD health from __WINDOW_DAYS__-day SMART trends.

You will think step by step, but your final output must be a single JSON object.

SMART attributes are grouped as:

ERROR / SPARE / WEAR (can indicate risk):
- r_5, r_181, r_182, r_183, r_184, r_187, r_195, r_197, r_199, r_180, r_177

WORKLOAD / LIFETIME (usage only, usually NOT risk by itself):
- r_9, r_12, r_241, r_242

Generic patterns like:
- "frequent bursts across most of the patch across all __WINDOW_DAYS__ days of the window"
- "a mix of stable, trending, and spiky attributes across all __WINDOW_DAYS__ days of the window"
are common in **both** healthy and failed drives and are **weak evidence**.

[NOVEL] means statistically unusual, but:
- [NOVEL] only on WORKLOAD / LIFETIME attributes (r_9, r_241, r_242, r_12) usually happens on healthy drives too.
- Treat [NOVEL] on error / spare / wear attributes as much more serious.

For each __WINDOW_DAYS__-day window, reason in this simple order:

1) Check if all attributes have the same kind of generic trend text
   (e.g., everything is "frequent bursts across most of the patch
   across all __WINDOW_DAYS__ days of the window") and the cross-attribute summary
   is also generic (e.g., "a mix of stable, trending, and spiky
   attributes across all __WINDOW_DAYS__ days of the window").

   - If YES, and there is NO [NOVEL] tag on any ERROR / SPARE / WEAR
     attribute, then directly decide: status = "HEALTHY"

2) If there are [NOVEL] segments, separate them into two groups:
   - Group A: [NOVEL] on WORKLOAD / LIFETIME only (r_9, r_12, r_241, r_242)
   - Group B: [NOVEL] on ERROR / SPARE / WEAR (r_5, r_181, r_182, r_183,
     r_184, r_187, r_195, r_197, r_199, r_180, r_177)

   - If only Group A is present (novelty only on r_9 / r_12 / r_241 / r_242)
     and all error/spare/wear attributes still look generic (no spikes,
     no clear worsening trend), then: status = "HEALTHY"

3) Only consider AT_RISK if Group B has at least one attribute with [NOVEL]
   or clearly worsening summary.

   For those ERROR / SPARE / WEAR attributes:
   - Look at the raw_stats if present:
       * non-zero values
       * clearly increasing last/ delta
       * activity near the end of the __WINDOW_DAYS__-day window

   - If there is at least one ERROR / SPARE / WEAR attribute that is clearly
     worsening and looks problematic, you may predict AT_RISK.
   - If this evidence is weak or ambiguous, stay with HEALTHY.

4) Time-to-failure (TTF), only if status = AT_RISK:
   - Use rough intuition:
       * If the problematic behaviour is very recent and sharp (mainly in
         the last 0–7 days), choose ttf_bucket = "<7".
       * If it is gradually worsening over much of the __WINDOW_DAYS__ days, choose
         ttf_bucket = "7-30".
       * If it looks risky but mostly earlier and less sharp, choose
         ttf_bucket = ">30".
   - ttf_days should be a rough integer consistent with the bucket.
   - If status = HEALTHY: ttf_days = null, ttf_bucket = "NONE".

5) Explanation and recommendations:
   - Keep them short and consistent with the decision rules above.
   - For HEALTHY: "LOW concern" and simple monitoring advice.
   - For AT_RISK: explain which ERROR / SPARE / WEAR attributes and
     patterns drove the decision and give 1–2 concrete next steps.

OUTPUT FORMAT:

After you finish reasoning, output ONLY a single JSON object:

{
  "status": "HEALTHY" or "AT_RISK",
  "concern_level": "LOW concern" | "MEDIUM concern" | "SERIOUS concern",
  "ttf_days": integer or null,
  "ttf_bucket": "<7" | "7-30" | ">30" | "NONE",
  "explanation": "short text",
  "recommendations": [
    "short actionable sentence 1",
    "short actionable sentence 2"
  ]
}

Rules:
- If status = "HEALTHY":
    - concern_level = "LOW concern"
    - ttf_days = null
    - ttf_bucket = "NONE"
- Never include backticks, labels, or any text outside the JSON.
"""


def build_system_prompt(window_days: int) -> str:
    return SYSTEM_PROMPT_TEMPLATE.replace("__WINDOW_DAYS__", str(window_days))





# ============================================================
# Utility: build text summary from prototypes
# ============================================================

def window_to_summary(
    idx: int,
    attr_protos: np.ndarray,
    attr_novel: np.ndarray,
    cross_protos: np.ndarray,
    cross_novel: np.ndarray,
    feature_names: np.ndarray,
    attr_phrases: np.ndarray,
    cross_phrases: np.ndarray,
    patch_len_attr: int,
    patch_len_cross: int,
    raw_window: np.ndarray = None,
) -> str:
    """
    Convert one window's prototype assignments into a human-readable
    description to feed to the LLM.

    Rules:
      - If ALL 6 patches for an attribute have the same phrase and same
        NOVEL flag, summarize as:
           "<phrase> across all 30 days of the window"
        (with "[NOVEL; ...]" attached if needed).

      - If ANY patch differs (phrase or NOVEL flag), list EVERY patch
        explicitly as:
           "days 0-5: <phrase>; days 5-10: <phrase>; ..."

      - For NOVEL attribute segments, if raw_window is available, attach:
           [NOVEL; raw_stats(...); raw_values=[...]]
        where raw_values are the actual SMART values over that segment.

      - Cross-attribute segments follow the same “all-same vs per-patch”
        rule, but we do NOT attach raw_values (no simple 1D series).
    """

    F, P_attr = attr_protos[idx].shape
    P_cross = cross_protos[idx].shape[0]
    total_days_attr = patch_len_attr * P_attr
    total_days_cross = patch_len_cross * P_cross

    lines: List[str] = []
    lines.append(f"Window index: {idx}")
    lines.append("")
    lines.append("Attribute-level patterns (per SMART attribute):")

    # ------------------------------------------------------------
    # Helper: build stats + raw_values string for an attribute
    # ------------------------------------------------------------
    def build_stats_and_values(start_day: int, end_day: int, f_idx_local: int) -> str:
        """
        Compute raw_stats and raw_values for attribute f_idx_local over
        [start_day, end_day), using raw_window (shape [T, F]).
        Returns a string like:
          "; raw_stats(min=..., max=..., mean=..., last=..., delta=...); raw_values=[...]"
        or "" if raw_window is None or invalid.
        """
        if raw_window is None:
            return ""

        s = start_day
        e = min(end_day, raw_window.shape[0])
        vals = raw_window[s:e, f_idx_local]
        if vals.size == 0:
            return ""

        vals = vals.astype(float)
        v_min = float(vals.min())
        v_max = float(vals.max())
        v_mean = float(vals.mean())
        v_last = float(vals[-1])
        v_first = float(vals[0])
        v_delta = float(v_last - v_first)

        # compact numeric formatting
        stats_part = (
            f"raw_stats(min={v_min:.3g}, max={v_max:.3g}, "
            f"mean={v_mean:.3g}, last={v_last:.3g}, delta={v_delta:.3g})"
        )

        # raw values (rounded) to let LLM inspect actual pattern
        vals_rounded = [round(float(x), 3) for x in vals]
        values_part = "raw_values=[" + ", ".join(str(x) for x in vals_rounded) + "]"

        return "; " + stats_part + "; " + values_part

    # ------------------------------------------------------------
    # Attribute-level summaries
    # ------------------------------------------------------------
    for f_idx in range(F):
        attr_name = str(feature_names[f_idx])

        # Build per-patch info: (patch_index, phrase, novel_flag)
        patches = []
        for p in range(P_attr):
            k = int(attr_protos[idx, f_idx, p])
            if k < 0:
                continue
            phrase = (
                attr_phrases[k]
                if 0 <= k < len(attr_phrases)
                else "unclassified pattern"
            )
            novel_flag = bool(attr_novel[idx, f_idx, p])
            patches.append((p, phrase, novel_flag))

        if not patches:
            continue

        # Check if ALL patches share the same (phrase, novel_flag) and cover [0, P_attr)
        first_phrase = patches[0][1]
        first_novel = bool(patches[0][2])
        all_same = (
            len(patches) == P_attr
            and patches[0][0] == 0
            and patches[-1][0] == P_attr - 1
            and all(
                (phrase == first_phrase and bool(novel) == first_novel)
                for (_, phrase, novel) in patches
            )
        )

        seg_strings: List[str] = []

        if all_same:
            # Single summary for full 30-day window
            seg = f"{first_phrase} across all {total_days_attr} days of the window"
            if first_novel:
                extra = build_stats_and_values(0, total_days_attr, f_idx)
                seg += f" [NOVEL{extra}]"
            seg_strings.append(seg)
        else:
            # At least one patch differs: show full per-patch breakdown
            for (p, phrase, novel_flag) in patches:
                start_day = p * patch_len_attr
                end_day = (p + 1) * patch_len_attr
                seg = f"days {start_day}-{end_day}: {phrase}"
                if novel_flag:
                    extra = build_stats_and_values(start_day, end_day, f_idx)
                    seg += f" [NOVEL{extra}]"
                seg_strings.append(seg)

        lines.append(f"- {attr_name}: " + "; ".join(seg_strings))

    # ------------------------------------------------------------
    # Cross-attribute summaries
    # ------------------------------------------------------------
    lines.append("")
    lines.append("Cross-attribute patterns over the whole feature set:")

    cross_patches = []
    for p in range(P_cross):
        k = int(cross_protos[idx, p])
        if k < 0:
            continue
        phrase = (
            cross_phrases[k]
            if 0 <= k < len(cross_phrases)
            else "unclassified pattern"
        )
        novel_flag = bool(cross_novel[idx, p])
        cross_patches.append((p, phrase, novel_flag))

    if cross_patches:
        first_phrase_c = cross_patches[0][1]
        first_novel_c = bool(cross_patches[0][2])
        all_same_c = (
            len(cross_patches) == P_cross
            and cross_patches[0][0] == 0
            and cross_patches[-1][0] == P_cross - 1
            and all(
                (phrase == first_phrase_c and bool(novel) == first_novel_c)
                for (_, phrase, novel) in cross_patches
            )
        )

        if all_same_c:
            # Single cross-attribute summary
            seg = f"{first_phrase_c} across all {total_days_cross} days of the window"
            if first_novel_c:
                seg += " [NOVEL]"
            lines.append(f"- {seg}")
        else:
            # Mixed behaviour: show each cross segment explicitly
            for (p, phrase, novel_flag) in cross_patches:
                start_day = p * patch_len_cross
                end_day = (p + 1) * patch_len_cross
                seg = f"segment days {start_day}-{end_day}: {phrase}"
                if novel_flag:
                    seg += " [NOVEL]"
                lines.append(f"- {seg}")

    return "\n".join(lines)



# ============================================================
# Utility: JSON parsing and metrics
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
    status_norm = status.strip().upper().replace("-", "_")
    if "AT_RISK" in status_norm or "RISK" in status_norm:
        return 1
    return 0


def compute_prf1(y_true: List[int], y_pred: List[int]) -> Tuple[float, float, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
    fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1


# ============================================================
# TTF bucket utilities
# ============================================================

# Fixed TTF buckets used by the LLM output schema
TTF_BUCKETS = ["<7", "7-30", ">30"]

# Midpoints (in days) for each bucket; used for bMAE and Cov_{±5}
TTF_BUCKET_MIDPOINTS = {
    "<7": 3.5,      # midpoint of [0,7)
    "7-30": 18.5,   # midpoint of [7,30]
    ">30": 45.0,    # assume failures beyond 30 days, adjust if needed
}


def ttf_to_bucket(ttf_val) -> str:
    """
    Map a scalar TTF (in days) to one of the discrete buckets.
    Returns "NONE" if ttf_val is invalid.
    """
    if ttf_val is None:
        return "NONE"
    try:
        t = float(ttf_val)
    except (TypeError, ValueError):
        return "NONE"
    if t < 0:
        return "NONE"
    if t < 7:
        return "<7"
    if t <= 30:
        return "7-30"
    return ">30"



# ============================================================
# Data loading helpers
# ============================================================

def load_status_labels(
    test_proto_npz,
    test_npz_path: Path,
) -> np.ndarray:
    """
    Try to load status labels (0=healthy, 1=failed) from:
      1) mb2_test_prototypes.npz, or
      2) processed/MB2_round1/test.npz
    using common key names.
    """
    candidate_keys = ["y", "labels", "y_fail", "label", "status", "y_status"]

    for k in candidate_keys:
        if k in test_proto_npz.files:
            print(f"Found status labels in mb2_test_prototypes.npz key '{k}'")
            return test_proto_npz[k]

    if test_npz_path.exists():
        base_npz = np.load(test_npz_path, allow_pickle=True)
        for k in candidate_keys:
            if k in base_npz.files:
                print(f"Found status labels in {test_npz_path} key '{k}'")
                return base_npz[k]

    raise RuntimeError(
        "Could not find status labels in mb2_test_prototypes.npz or test.npz.\n"
        "Please ensure one of the keys "
        f"{candidate_keys} is present."
    )

def get_raw_window(raw_X, idx, F, P_attr, patch_len_attr):
    """
    Return raw window as [T, F] for the given index, handling (T,F) or (F,T)
    layouts. If shapes don't match expectations, return None.
    """
    w = raw_X[idx]
    if w.ndim != 2:
        print(f"WARNING: raw window {idx} has ndim={w.ndim}, expected 2; skipping raw stats.")
        return None

    T_expected = patch_len_attr * P_attr

    if w.shape == (T_expected, F):
        return w
    if w.shape == (F, T_expected):
        return w.T

    print(f"WARNING: raw window {idx} shape={w.shape} does not match "
          f"expected (T={T_expected}, F={F}) or (F, T); skipping raw stats.")
    return None


# ============================================================
# Main evaluation loop
# ============================================================

def infer_artifact_prefix(dataset_name: str) -> str:
    dataset_name = dataset_name.strip().upper()
    if dataset_name not in {"MB1", "MB2"}:
        raise ValueError("--dataset-name must be one of: MB1, MB2")
    return dataset_name.lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="MB2",
                        help="Dataset/model name, e.g. MB1 or MB2.")
    parser.add_argument("--round", type=int, default=1,
                        help="Temporal split round number (default: 1)")
    parser.add_argument("--artifact-root", type=str, default="data/artifacts",
                        help="Root directory containing dataset round artifact folders.")
    parser.add_argument("--processed-root", type=str, default="data/processed",
                        help="Root directory containing dataset round processed folders.")
    parser.add_argument("--artifact-dir", type=str, default=None,
                        help="Optional explicit artifact directory for this run.")
    parser.add_argument("--processed-dir", type=str, default=None,
                        help="Optional explicit processed directory for this run.")
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
    parser.add_argument("--model-name", type=str, default="Llama-3.1-8B-Instruct",
                        help="vLLM served model name")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1",
                        help="OpenAI-compatible base URL for vLLM")
    parser.add_argument("--api-key", type=str, default="EMPTY",
                        help="API key for vLLM (often 'EMPTY')")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens to generate (default: 512)")
    parser.add_argument("--output-jsonl", type=str, default=None,
                        help="Optional JSONL path for per-window predictions.")
    parser.add_argument("--output-metrics-json", type=str, default=None,
                        help="Optional JSON path for aggregate status/TTF metrics.")
    parser.add_argument("--output-tp-csv", type=str, default=None,
                        help="Optional CSV path for true-positive TTF details.")
    args = parser.parse_args()

    dataset_name = args.dataset_name.strip().upper()
    artifact_prefix = infer_artifact_prefix(dataset_name)
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
        print("WARNING: feature_names differ between prototypes and test_prototypes; "
              "using test_prototypes feature order.")
    feature_names = feature_names_test

    N, F, P_attr = attr_protos.shape
    print(f"Test windows: N={N}, F={F}, P_attr={P_attr}")

    # Load raw windows for statistics on NOVEL patches
    if base_test_path.exists():
        print(f"Loading raw test windows from {base_test_path} ...")
        base_npz = np.load(base_test_path, allow_pickle=True)
        raw_X = base_npz["X"]
        print(f"Raw X shape: {raw_X.shape}")

        # Ground-truth TTF (in days) for each window, used for TTF metrics
        if "ttf" in base_npz.files:
            ttf = base_npz["ttf"]
            print(f"Loaded TTF vector with shape: {ttf.shape}")
        else:
            print("WARNING: 'ttf' not found in test.npz; TTF metrics will be disabled.")
            ttf = None

        if "features" in base_npz.files:
            raw_features = base_npz["features"]
            if not np.array_equal(feature_names, raw_features):
                print("WARNING: feature_names differ between test_prototypes and raw X; "
                      "assuming positional alignment.")
    else:
        print(f"WARNING: {base_test_path} not found; raw stats for NOVEL patches will be disabled.")
        raw_X = None
        ttf = None



    # Load status labels
    y_status = load_status_labels(test_npz, base_test_path)
    y_status = np.asarray(y_status).astype(int)
    assert y_status.shape[0] == N, "Label length mismatch."

    # Sample all selected failures first, then randomly add healthy windows
    # according to the requested healthy_per_fail ratio.
    fail_idx = np.where(y_status == 1)[0]
    healthy_idx = np.where(y_status == 0)[0]

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


    # vLLM client
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    prediction_records = []

    tp_records = []

    for rank, idx in enumerate(selected, start=1):
        # print("=" * 80)
        print(f"Window {rank}/{total} (global index {idx})")
        gt = int(y_status[idx])
        # print(f"Ground truth status: {'FAILED/AT_RISK (1)' if gt == 1 else 'HEALTHY (0)'}")

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
        if raw_window is not None:
            window_days = int(raw_window.shape[0])
        else:
            window_days = int(max(attr_protos[idx].shape[1] * patch_len_attr, cross_protos[idx].shape[0] * patch_len_cross))
        # print(summary_text)

        user_message = (
            f"Think step by step. You are given prototype-based trend summaries for one SSD over a {window_days}-day "
            f"window. Use them to decide whether the drive is HEALTHY or AT_RISK, "
            "and only if AT_RISK, estimate time-to-failure as described.\n\n"
            "SMART prototype summary:\n"
            + summary_text
        )
        # print(user_message)

        response = client.chat.completions.create(
            model=args.model_name,
            messages=[
                {"role": "system", "content": build_system_prompt(window_days)},
                {"role": "user", "content": user_message},
            ],
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
        # print("___________________________", status_str)
        concern_level = str(obj.get("concern_level", "LOW concern"))
        ttf_days = obj.get("ttf_days", None)
        ttf_bucket = str(obj.get("ttf_bucket", "NONE"))
        explanation = str(obj.get("explanation", ""))
        recommendations = obj.get("recommendations", [])

        y_true_all.append(gt)
        pred_status_int = status_to_int(status_str)
        y_pred_all.append(pred_status_int)
        prediction_records.append(
            {
                "index": int(idx),
                "dataset_name": dataset_name,
                "round": args.round,
                "summary": summary_text,
                "true_status": int(gt),
                "pred_status": int(pred_status_int),
                "true_ttf_days": None if ttf is None else float(ttf[idx]),
                "pred_ttf_days": ttf_days,
                "pred_ttf_bucket": ttf_bucket,
                "concern_level": concern_level,
                "explanation": explanation,
                "recommendations": recommendations,
                "raw_response": obj,
            }
        )

        # --- Collect TTF info for TRUE POSITIVES only (y=1 and pred=AT_RISK) ---
        if (
            ttf is not None
            and gt == 1
            and pred_status_int == 1
        ):
            # Ground-truth TTF in days for this window
            true_ttf = float(ttf[idx])

            # Normalise predicted bucket
            pred_bucket = ttf_bucket.strip()
            if pred_bucket not in TTF_BUCKET_MIDPOINTS:
                pred_mid = None
            else:
                pred_mid = TTF_BUCKET_MIDPOINTS[pred_bucket]

            # Normalise ttf_days to int or None
            if isinstance(ttf_days, (int, float)):
                ttf_days_norm = int(ttf_days)
            else:
                try:
                    ttf_days_norm = int(ttf_days)
                except Exception:
                    ttf_days_norm = None

            # Flatten recommendations for CSV
            if isinstance(recommendations, list):
                rec_str = " || ".join(str(r) for r in recommendations)
            else:
                rec_str = str(recommendations)

            tp_records.append(
                {
                    "idx": int(idx),
                    "status_true": int(gt),
                    "status_pred": status_str,
                    "ttf_true": true_ttf,
                    "ttf_days_pred": ttf_days_norm,
                    "ttf_bucket_pred": pred_bucket,
                    "ttf_bucket_midpoint": pred_mid,
                    "explanation": explanation,
                    "recommendations": rec_str,
                }
            )


        # print("\nParsed prediction:")
        # print(f"  status        : {status_str}")
        # print(f"  concern_level : {concern_level}")
        # print(f"  ttf_days      : {ttf_days}")
        # print(f"  ttf_bucket    : {ttf_bucket}")
        # print(f"  explanation   : {explanation}")
        # print(f"  recommendations:")
        # if isinstance(recommendations, list):
        #     for r in recommendations:
        #         print(f"    - {r}")
        # else:
        #     print(f"    - {recommendations}")

    # --- Overall metrics ---
        # --- Overall metrics ---
    if y_true_all and y_pred_all:
        # convert to numpy arrays
        y_true_arr = np.asarray(y_true_all, dtype=int)
        y_pred_arr = np.asarray(y_pred_all, dtype=int)

        # confusion matrix components
        tp = int(((y_true_arr == 1) & (y_pred_arr == 1)).sum())
        fp = int(((y_true_arr == 0) & (y_pred_arr == 1)).sum())
        fn = int(((y_true_arr == 1) & (y_pred_arr == 0)).sum())
        tn = int(((y_true_arr == 0) & (y_pred_arr == 0)).sum())

        # precision / recall / F1 using your helper
        precision, recall, f1 = compute_prf1(y_true_all, y_pred_all)

        print("\n=== Confusion matrix (sampled windows) ===")
        print(f"TP (true positives,  y=1, pred=1): {tp}")
        print(f"FP (false positives, y=0, pred=1): {fp}")
        print(f"FN (false negatives, y=1, pred=0): {fn}")
        print(f"TN (true negatives,  y=0, pred=0): {tn}")

        print("\n=== Overall classification metrics on sampled windows ===")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")

        # --- TTF metrics and CSV export on true positives only ---
        if tp_records:
            # Keep only records with a valid predicted bucket and midpoint
            valid_tp = [
                r for r in tp_records
                if r["ttf_bucket_pred"] in TTF_BUCKET_MIDPOINTS
                and r["ttf_bucket_midpoint"] is not None
            ]

            if valid_tp:
                true_ttf_arr = np.array([r["ttf_true"] for r in valid_tp], dtype=float)
                pred_bucket_arr = np.array([r["ttf_bucket_pred"] for r in valid_tp], dtype=object)

                # Map ground-truth TTF to buckets
                true_bucket_arr = np.array(
                    [ttf_to_bucket(t) for t in true_ttf_arr],
                    dtype=object,
                )
                mask = true_bucket_arr != "NONE"
                true_ttf_arr = true_ttf_arr[mask]
                pred_bucket_arr = pred_bucket_arr[mask]
                true_bucket_arr = true_bucket_arr[mask]

                if true_bucket_arr.size == 0:
                    print("\nNo valid TTF labels among true positives; skipping TTF metrics.")
                else:
                    # Macro-averaged F1 across TTF buckets (unweighted mean of per-bucket F1)
                    per_bucket_f1 = []
                    used_buckets = 0
                    for b in TTF_BUCKETS:
                        tp_b = int(((true_bucket_arr == b) & (pred_bucket_arr == b)).sum())
                        fp_b = int(((true_bucket_arr != b) & (pred_bucket_arr == b)).sum())
                        fn_b = int(((true_bucket_arr == b) & (pred_bucket_arr != b)).sum())

                        # Skip buckets with no true examples
                        if tp_b + fn_b == 0:
                            continue

                        if tp_b == 0:
                            f1_b = 0.0
                        else:
                            prec_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0
                            rec_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
                            f1_b = (
                                2.0 * prec_b * rec_b / (prec_b + rec_b)
                                if (prec_b + rec_b) > 0
                                else 0.0
                            )
                        per_bucket_f1.append(f1_b)
                        used_buckets += 1

                    if used_buckets > 0:
                        macro_f1_ttf = float(np.mean(per_bucket_f1))
                    else:
                        macro_f1_ttf = float("nan")

                    # Bucketed MAE (bMAE) and coverage Cov_{±5}
                    pred_mid_arr = np.array(
                        [TTF_BUCKET_MIDPOINTS[b] for b in pred_bucket_arr],
                        dtype=float,
                    )
                    abs_err = np.abs(true_ttf_arr - pred_mid_arr)
                    bmae = float(abs_err.mean())
                    cov_pm5 = float((abs_err <= 5.0).mean())

                    print("\n=== TTF metrics on true positives (y=1, pred=1) ===")
                    print(f"TTF macro-F1 over buckets : {macro_f1_ttf:.4f}")
                    print(f"TTF bucketed MAE (bMAE)   : {bmae:.4f}")
                    print(f"TTF Cov_±5                : {cov_pm5:.4f}")
            else:
                print("\nNo true positives with valid TTF bucket predictions; skipping TTF metrics.")

            # Write all true-positive records (including ones with invalid bucket) to CSV
            safe_model_name = args.model_name.replace("/", "_").replace(" ", "_")
            tp_csv_path = Path(args.output_tp_csv) if args.output_tp_csv else artifact_root / f"{artifact_prefix}_eval_tp_details_{safe_model_name}.csv"
            tp_csv_path.parent.mkdir(parents=True, exist_ok=True)

            fieldnames = [
                "idx",
                "status_true",
                "status_pred",
                "ttf_true",
                "ttf_days_pred",
                "ttf_bucket_pred",
                "ttf_bucket_midpoint",
                "explanation",
                "recommendations",
            ]
            with tp_csv_path.open("w", newline="") as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
                writer.writeheader()
                for r in tp_records:
                    writer.writerow(r)

            print(f"\nWrote {len(tp_records)} true-positive records to {tp_csv_path}")
        else:
            print("\nNo true positives; skipping TTF metrics and CSV export.")

        if args.output_jsonl:
            output_jsonl = Path(args.output_jsonl)
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with output_jsonl.open("w", encoding="utf-8") as f_out:
                for record in prediction_records:
                    f_out.write(json.dumps(record) + "\n")
            print(f"Wrote per-window predictions to {output_jsonl}")

        if args.output_metrics_json:
            ttf_payload = None
            if tp_records:
                valid_tp = [
                    r for r in tp_records
                    if r["ttf_bucket_pred"] in TTF_BUCKET_MIDPOINTS
                    and r["ttf_bucket_midpoint"] is not None
                    and ttf_to_bucket(r["ttf_true"]) != "NONE"
                ]
                if valid_tp:
                    true_ttf_arr = np.array([r["ttf_true"] for r in valid_tp], dtype=float)
                    pred_bucket_arr = np.array([r["ttf_bucket_pred"] for r in valid_tp], dtype=object)
                    true_bucket_arr = np.array([ttf_to_bucket(t) for t in true_ttf_arr], dtype=object)
                    per_bucket_f1 = []
                    for b in TTF_BUCKETS:
                        tp_b = int(((true_bucket_arr == b) & (pred_bucket_arr == b)).sum())
                        fp_b = int(((true_bucket_arr != b) & (pred_bucket_arr == b)).sum())
                        fn_b = int(((true_bucket_arr == b) & (pred_bucket_arr != b)).sum())
                        if tp_b + fn_b == 0:
                            continue
                        prec_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0
                        rec_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
                        f1_b = (
                            2.0 * prec_b * rec_b / (prec_b + rec_b)
                            if (prec_b + rec_b) > 0 else 0.0
                        )
                        per_bucket_f1.append(f1_b)
                    pred_mid_arr = np.array(
                        [TTF_BUCKET_MIDPOINTS[b] for b in pred_bucket_arr],
                        dtype=float,
                    )
                    abs_err = np.abs(true_ttf_arr - pred_mid_arr)
                    ttf_payload = {
                        "n_true_positives_scored": len(valid_tp),
                        "macro_f1": float(np.mean(per_bucket_f1)) if per_bucket_f1 else None,
                        "bmae": float(abs_err.mean()),
                        "cov_pm5": float((abs_err <= 5.0).mean()),
                    }

            metrics_payload = {
                "dataset_name": dataset_name,
                "round": args.round,
                "method": "SMARTTalk",
                "model_name": args.model_name,
                "n_examples_evaluated": len(y_pred_all),
                "classification": {
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f05": f05_score(precision, recall),
                    "fpr": safe_div(fp, fp + tn),
                    "fnr": safe_div(fn, tp + fn),
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
                "ttf": ttf_payload,
            }
            output_metrics_json = Path(args.output_metrics_json)
            output_metrics_json.parent.mkdir(parents=True, exist_ok=True)
            output_metrics_json.write_text(json.dumps(metrics_payload, indent=2) + "\n")
            print(f"Wrote aggregate metrics to {output_metrics_json}")



    else:
        print("No valid predictions collected; cannot compute metrics.")



if __name__ == "__main__":
    main()
