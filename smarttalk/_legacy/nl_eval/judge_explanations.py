#!/usr/bin/env python3
"""
Score SMARTTalk explanations and recommendations with an LLM-as-a-judge.

This script accepts either:
- SMARTTalk prediction JSONL emitted by `code/core/llm_eval.py`, or
- explanation CSV files emitted by `code/nl_eval/exp_rec_generation.py`.

The paper-facing Table 7 path uses the SMARTTalk prediction JSONL so the judge
sees the method's predicted status and TTF bucket together with the same SMART
summary the method used.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from openai import OpenAI


JUDGE_PROMPT_TEMPLATE = """
You are an impartial expert on SSD reliability and SMART-based failure prediction.

You will be given:
1. A textual summary of a 30-day SMART window for one SSD.
2. A method's predicted status and time-to-failure bucket.
3. The method's natural-language explanation.
4. The method's recommended actions for the operator.

Your job is to act purely as a judge for this method's outputs. You must not
re-predict status or TTF yourself. Treat the given predicted status and TTF as
the method's claim, and evaluate how well its explanation and recommendations
match that claim and the SMART summary.

Produce two integer 1-5 scores:
1) ExpScore: how correct and faithful the explanation is to the prediction and
the SMART trends.
2) RecScore: how useful and safe the recommended actions are for a cloud operator.

Be strict but fair. Do not change the prediction; only judge the explanation and
recommendation quality relative to the given summary and prediction.

You must respond only with a single JSON object, no extra text.

[SMART_SUMMARY]
-------------
SMART summary (30-day window):
{SMART_SUMMARY}

Method's prediction:
- status: {PRED_STATUS}
- time_to_failure_bucket: {PRED_TTF_BUCKET}

Method's explanation:
{EXPLANATION}

Method's recommended actions:
{RECOMMENDATIONS}
-------------

Return only JSON with:
- "ExpScore": integer 1-5
- "RecScore": integer 1-5
- "exp_rationale": short text
- "rec_rationale": short text
""".strip()


def extract_json(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in judge output.")
    return json.loads(text[start : end + 1])


def normalize_status(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip().upper().replace("-", "_")
        if text in {"1", "AT_RISK", "FAILED"} or "RISK" in text or "FAIL" in text:
            return "AT_RISK"
        return "HEALTHY"
    return "AT_RISK" if int(value) == 1 else "HEALTHY"


def normalize_bucket(value: Any) -> str:
    text = str(value).strip().upper()
    if text in {"NONE", "NO_FAILURE", "NULL", ""}:
        return "NO_FAILURE"
    return str(value).strip()


def load_input_records(path: Path, filter_mode: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if filter_mode == "true_positive":
                if int(row.get("true_status", 0)) != 1 or int(row.get("pred_status", 0)) != 1:
                    continue
            records.append(
                {
                    "index": int(row["index"]),
                    "dataset_name": row.get("dataset_name", ""),
                    "round": row.get("round", ""),
                    "summary": str(row["summary"]),
                    "pred_status_text": normalize_status(row.get("pred_status", 0)),
                    "pred_ttf_bucket": normalize_bucket(row.get("pred_ttf_bucket", "NONE")),
                    "explanation": str(row.get("explanation", "")),
                    "recommendations_text": " || ".join(str(x) for x in row.get("recommendations", []))
                    if isinstance(row.get("recommendations", []), list)
                    else str(row.get("recommendations", "")),
                }
            )
        return records

    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                records.append(
                    {
                        "index": int(row["idx"]),
                        "dataset_name": "",
                        "round": "",
                        "summary": str(row["summary"]),
                        "pred_status_text": normalize_status(row.get("status_label", "FAILED")),
                        "pred_ttf_bucket": normalize_bucket(row.get("ttf_label_bucket", "NONE")),
                        "explanation": str(row.get("explanation", "")),
                        "recommendations_text": row.get("recommendations_json", "[]"),
                    }
                )
        return records

    raise ValueError("Unsupported input format. Use .jsonl or .csv.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the supplementary Figure 4 judge prompt over SMARTTalk explanation outputs.",
    )
    parser.add_argument("--input-path", type=str, required=True,
                        help="Prediction JSONL from code/core/llm_eval.py or CSV from exp_rec_generation.py.")
    parser.add_argument("--output-csv", type=str, required=True,
                        help="Path to write per-window judge scores.")
    parser.add_argument("--output-metrics-json", type=str, default=None,
                        help="Optional path to write aggregate ExpScore/RecScore means.")
    parser.add_argument("--filter-mode", type=str, default="true_positive",
                        choices=["true_positive", "all"],
                        help="Which records to score when input is prediction JSONL.")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Judge model name. The paper used an external GPT-5.1 Thinking judge.")
    parser.add_argument("--base-url", type=str, default="https://api.openai.com/v1",
                        help="OpenAI-compatible judge endpoint.")
    parser.add_argument("--api-key", type=str, default="",
                        help="API key for the judge endpoint or set OPENAI_API_KEY.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Judge sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=400,
                        help="Max completion tokens for the judge.")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    records = load_input_records(input_path, args.filter_mode)
    if not records:
        raise RuntimeError(f"No records available for judging in {input_path}.")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("No API key provided. Pass --api-key or set OPENAI_API_KEY.")

    client = OpenAI(base_url=args.base_url, api_key=api_key)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    exp_scores: List[int] = []
    rec_scores: List[int] = []
    for row in records:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            SMART_SUMMARY=row["summary"],
            PRED_STATUS=row["pred_status_text"],
            PRED_TTF_BUCKET=row["pred_ttf_bucket"],
            EXPLANATION=row["explanation"],
            RECOMMENDATIONS=row["recommendations_text"],
        )
        response = client.chat.completions.create(
            model=args.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        raw_output = response.choices[0].message.content or ""
        try:
            obj = extract_json(raw_output)
        except Exception as exc:
            raise RuntimeError(
                f"Judge output for record {row['index']} was not valid JSON. "
                "This packaged evaluator does not use free-text fallbacks."
            ) from exc

        exp_score = int(obj["ExpScore"])
        rec_score = int(obj["RecScore"])
        exp_scores.append(exp_score)
        rec_scores.append(rec_score)
        rows.append(
            {
                "index": row["index"],
                "dataset_name": row.get("dataset_name", ""),
                "round": row.get("round", ""),
                "pred_status": row["pred_status_text"],
                "pred_ttf_bucket": row["pred_ttf_bucket"],
                "exp_score": exp_score,
                "rec_score": rec_score,
                "exp_rationale": obj.get("exp_rationale", ""),
                "rec_rationale": obj.get("rec_rationale", ""),
                "raw_output": raw_output,
            }
        )

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote judge scores to {output_csv}")

    if args.output_metrics_json:
        payload = {
            "n_examples_scored": len(rows),
            "exp_score_mean": sum(exp_scores) / len(exp_scores),
            "rec_score_mean": sum(rec_scores) / len(rec_scores),
            "judge_model_name": args.model_name,
        }
        output_metrics_json = Path(args.output_metrics_json)
        output_metrics_json.parent.mkdir(parents=True, exist_ok=True)
        output_metrics_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote judge aggregate metrics to {output_metrics_json}")


if __name__ == "__main__":
    main()
