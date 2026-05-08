"""Seed helpers used by artifact wrappers and tests."""

from __future__ import annotations

import random

import numpy as np


def set_basic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
