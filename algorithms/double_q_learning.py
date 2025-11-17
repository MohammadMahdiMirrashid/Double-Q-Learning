"""Double Q-Learning agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class TrainingHistory:
    episode: List[int]
    reward: List[float]
    epsilon: List[float]


class DoubleQLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        min_epsilon: float = 0.01,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.q1 = np.zeros((n_states, n_actions), dtype=np.float32)
        self.q2 = np.zeros((n_states, n_actions), dtype=np.float32)

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_values = self.q1[state] + self.q2[state]
        return int(np.argmax(q_values))

    def _update_table(self, table_a, table_b, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            best_action = int(np.argmax(table_a[next_state]))
            target = reward + self.gamma * table_b[next_state, best_action]
        table_a[state, action] += self.lr * (target - table_a[state, action])

    def update(self, state, action, reward, next_state, done):
        if np.random.random() < 0.5:
            self._update_table(self.q1, self.q2, state, action, reward, next_state, done)
        else:
            self._update_table(self.q2, self.q1, state, action, reward, next_state, done)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def train(self, env, episodes: int = 1000, max_steps: int = 200) -> TrainingHistory:
        history = TrainingHistory(episode=[], reward=[], epsilon=[])
        for ep in range(1, episodes + 1):
            state = env.reset()
            total_reward = 0.0
            for _ in range(max_steps):
                action = self.select_action(state)
                step = env.step(action)
                self.update(state, action, step.reward, step.state, step.done)
                state = step.state
                total_reward += step.reward
                if step.done:
                    break
            self.decay_epsilon()
            history.episode.append(ep)
            history.reward.append(total_reward)
            history.epsilon.append(self.epsilon)
        return history

    def greedy_policy(self):
        def policy(state: int) -> int:
            q_values = self.q1[state] + self.q2[state]
            return int(np.argmax(q_values))

        return policy
