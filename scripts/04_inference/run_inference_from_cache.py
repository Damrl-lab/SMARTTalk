#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cache_dir = root / "artifacts" / "cached_predictions"
    print(f"Cached inference assets available under: {cache_dir}")
    for path in sorted(cache_dir.rglob("*")):
        if path.is_file():
            print(path.relative_to(root))


if __name__ == "__main__":
    main()
