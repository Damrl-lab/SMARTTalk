"""Lightweight parser helpers for structured LLM outputs."""

from __future__ import annotations

import json
from typing import Any, Dict


def extract_first_json_block(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start:end + 1])


def normalize_status(status: str) -> int:
    text = status.strip().upper().replace("-", "_")
    return 1 if "RISK" in text or "FAIL" in text else 0
