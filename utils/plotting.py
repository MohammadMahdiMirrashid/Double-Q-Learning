"""Plotting helpers for training curves."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np


def _moving_average(values: Iterable[float], window: int = 50) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size < window:
        window = max(1, values.size)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_rewards(history: Mapping[str, Iterable[float]], window: int = 50, title: str = "Episode Rewards", save_path: str | Path | None = None) -> None:
    episodes = np.asarray(history["episode"], dtype=np.int32)
    rewards = np.asarray(history["reward"], dtype=np.float32)
    ma = _moving_average(rewards, window)
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, rewards, alpha=0.3, label="reward")
    plt.plot(episodes[window - 1 :], ma, label=f"{window}-ep moving avg", color="tab:orange")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_epsilon(history: Mapping[str, Iterable[float]], save_path: str | Path | None = None) -> None:
    episodes = history["episode"]
    epsilon = history["epsilon"]
    plt.figure(figsize=(6, 3))
    plt.plot(episodes, epsilon, color="tab:green")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Exploration schedule")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
