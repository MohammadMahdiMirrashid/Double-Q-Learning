"""CLI for running Q-Learning vs Double Q-Learning experiments."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from algorithms.double_q_learning import DoubleQLearningAgent
from algorithms.q_learning import QLearningAgent
from environment.env_wrapper import make_env
from utils.helpers import set_global_seed, write_history_csv
from utils import plotting


ALGO_REGISTRY = {
    "q": QLearningAgent,
    "double_q": DoubleQLearningAgent,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tabular RL experiments.")
    parser.add_argument("--env-id", default="FrozenLake-v1", help="Gymnasium environment id")
    parser.add_argument("--algo", choices=ALGO_REGISTRY.keys(), default="double_q")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--min-epsilon", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-path", default="results/logs/training.csv")
    parser.add_argument("--reward-plot", default="results/plots/rewards.png")
    parser.add_argument("--epsilon-plot", default="results/plots/epsilon.png")
    parser.add_argument("--render-mode", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    env = make_env(args.env_id, seed=args.seed, render_mode=args.render_mode)

    agent_cls = ALGO_REGISTRY[args.algo]
    agent = agent_cls(
        n_states=env.n_states,
        n_actions=env.n_actions,
        lr=args.lr,
        gamma=args.gamma,
        epsilon=args.epsilon,
        min_epsilon=args.min_epsilon,
        epsilon_decay=args.epsilon_decay,
    )

    history = agent.train(env, episodes=args.episodes, max_steps=args.max_steps)
    hist_dict = asdict(history)

    write_history_csv(hist_dict, args.log_path)
    plotting.plot_rewards(hist_dict, title=f"{args.algo} on {args.env_id}", save_path=args.reward_plot)
    plotting.plot_epsilon(hist_dict, save_path=args.epsilon_plot)

    last_avg = sum(hist_dict["reward"][-100:]) / min(100, len(hist_dict["reward"]))
    print(f"Finished {args.episodes} episodes. Last-100 avg reward: {last_avg:.3f}")
    print(f"Logs saved to {Path(args.log_path).resolve()}")


if __name__ == "__main__":
    main()
