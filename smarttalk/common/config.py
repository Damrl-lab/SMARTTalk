"""Configuration loading helpers for CLI wrappers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute() and not path.exists():
        repo_relative = PACKAGE_ROOT / path
        if repo_relative.exists():
            path = repo_relative
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            return json.load(handle)
        return yaml.safe_load(handle)


def add_config_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=str, required=True, help="Path to YAML or JSON config.")
    return parser
