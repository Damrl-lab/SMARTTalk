"""Lightweight parser helpers for structured LLM outputs."""

from __future__ import annotations

import json
import re
from typing import Any, Dict


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _iter_json_candidates(text: str):
    for start in range(len(text)):
        if text[start] != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for end in range(start, len(text)):
            ch = text[end]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : end + 1]
                    break


def extract_first_json_block(text: str) -> Dict[str, Any]:
    cleaned = _strip_markdown_fences(text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    for candidate in _iter_json_candidates(cleaned):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    raise ValueError("No JSON object found in model output.")


def normalize_status(status: str) -> int:
    text = status.strip().upper().replace("-", "_")
    return 1 if "RISK" in text or "FAIL" in text else 0


def infer_status_label(text: str) -> str | None:
    normalized = text.strip().upper().replace("-", "_")
    if "AT_RISK" in normalized or "RISK" in normalized or "FAILED" in normalized:
        return "AT_RISK"
    if "HEALTHY" in normalized:
        return "HEALTHY"
    return None


def infer_ttf_bucket(text: str) -> str:
    normalized = text.strip().upper()
    if "<7" in normalized:
        return "<7"
    if "7-30" in normalized:
        return "7-30"
    if ">30" in normalized:
        return ">30"

    if re.search(r"\bWITHIN\s+7\s+DAYS?\b", normalized):
        return "<7"
    if re.search(r"\b7\s*(TO|-)\s*30\s+DAYS?\b", normalized):
        return "7-30"
    if re.search(r"\b(MORE THAN|OVER|GREATER THAN)\s+30\s+DAYS?\b", normalized):
        return ">30"
    return "NONE"


def infer_ttf_days(text: str) -> int | None:
    match = re.search(r"\b(\d{1,3})\s*DAYS?\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def best_effort_prediction_payload(text: str) -> Dict[str, Any] | None:
    status = infer_status_label(text)
    if status is None:
        return None

    if status == "HEALTHY":
        return {
            "status": "HEALTHY",
            "concern_level": "LOW concern",
            "ttf_days": None,
            "ttf_bucket": "NONE",
            "explanation": text.strip(),
            "recommendations": [],
        }

    bucket = infer_ttf_bucket(text)
    return {
        "status": "AT_RISK",
        "concern_level": "MEDIUM concern",
        "ttf_days": infer_ttf_days(text),
        "ttf_bucket": bucket if bucket != "NONE" else "7-30",
        "explanation": text.strip(),
        "recommendations": [],
    }
