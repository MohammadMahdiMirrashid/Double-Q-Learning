"""Utility helpers for experiments."""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def write_history_csv(history: Mapping[str, Iterable], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = zip(*[history[key] for key in history])
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(history.keys())
        for row in rows:
            writer.writerow(row)


def moving_average(values: Iterable[float], window: int) -> np.ndarray:
    values = np.asarray(list(values), dtype=np.float32)
    if values.size == 0:
        return values
    if values.size < window:
        window = values.size
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")
