"""Simple user-facing logging helpers."""

from __future__ import annotations


def log_step(message: str) -> None:
    print(f"[SMARTTalk] {message}")
