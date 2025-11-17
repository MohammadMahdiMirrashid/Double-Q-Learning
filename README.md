## Double Q-Learning Playground

This project provides a minimal-yet-complete reference implementation of tabular Q-Learning and Double Q-Learning using the Gymnasium API. It is organised to make it easy to run experiments, create plots, and explore the behaviour of both algorithms in classic control environments such as `FrozenLake-v1` or `CliffWalking-v0`.

### Project Layout

```
double_q_learning/
├── environment/             # Environment helpers & wrappers
├── algorithms/              # Q-Learning + Double Q-Learning agents
├── utils/                   # Plotting + misc helpers
├── experiments/             # Reproducible experiment scripts
├── notebooks/               # Interactive demos
└── results/                 # Auto-created outputs (plots + logs)
```

### Getting Started

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # On Windows use: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run a quick experiment**
   ```bash
   python experiments/run_experiments.py \
       --env-id FrozenLake-v1 \
       --algo double_q \
       --episodes 2000
   ```
   The script prints rolling returns to stdout and stores raw episode rewards under `results/logs/`.

3. **Explore interactively**

   Open `notebooks/demo.ipynb` to visualise training curves and compare Q-Learning vs Double Q-Learning behaviour.

### Key Features

- Clean, tabular implementations with epsilon-greedy exploration and configurable decay schedules.
- Support for deterministic or stochastic Gymnasium environments via an optional wrapper.
- Built-in plotting utilities for cumulative rewards, moving averages, and epsilon schedules.
- Reusable experiment runner for benchmarking and logging results.

### Extending

- Swap in a different Gymnasium environment by editing `experiments/run_experiments.py` arguments.
- Tweak hyperparameters (learning rate, discount factor, exploration decay) by passing CLI flags or modifying the notebook.
- Implement additional algorithms (e.g., SARSA, Expected SARSA) alongside the provided structure.

Happy experimenting!
