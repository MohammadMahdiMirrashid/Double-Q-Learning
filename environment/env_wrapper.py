"""Simple wrappers around Gymnasium environments for tabular agents."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import gymnasium as gym


@dataclass
class StepResult:
    state: int
    reward: float
    done: bool
    info: Dict[str, Any]


class DiscreteEnvWrapper:
    """Minimal wrapper that flattens Gymnasium return signature for tabular RL."""

    def __init__(self, env: gym.Env):
        if not isinstance(env.observation_space, gym.spaces.Discrete):
            raise TypeError("Only discrete observation spaces are supported in this reference impl.")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("Only discrete action spaces are supported in this reference impl.")
        self.env = env
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n

    def reset(self, *, seed: int | None = None) -> int:
        obs, _ = self.env.reset(seed=seed)
        return int(obs)

    def step(self, action: int) -> StepResult:
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        return StepResult(state=int(obs), reward=float(reward), done=done, info=info)

    def close(self) -> None:
        self.env.close()


def make_env(env_id: str = "FrozenLake-v1", seed: int | None = None, render_mode: str | None = None, **env_kwargs: Any) -> DiscreteEnvWrapper:
    """Instantiate a Gymnasium environment and wrap it for tabular algorithms."""

    env = gym.make(env_id, render_mode=render_mode, **env_kwargs)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return DiscreteEnvWrapper(env)


def evaluate_policy(env: DiscreteEnvWrapper, policy, episodes: int = 10) -> float:
    """Roll out a greedy policy and return its mean episodic reward."""

    total_reward = 0.0
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = policy(state)
            step = env.step(action)
            total_reward += step.reward
            done = step.done
    return total_reward / episodes
