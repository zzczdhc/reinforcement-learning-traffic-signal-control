# Reinforcement Learning for Adaptive Traffic Signal Control

This repository is a course-project starter for adaptive traffic signal control at a single intersection under stationary and nonstationary demand.

## Status

Implemented now:

- Gymnasium-compatible single-intersection simulator with stochastic arrivals
- minimum-green enforcement, true yellow transitions, and invalid-switch tracking
- three heuristic baselines: fixed-cycle, queue-threshold, max-pressure
- DQN training loop with replay buffer and target network
- ablation runner for reward, state, and switch-penalty studies
- automatic figure generation from aggregated experiment outputs
- JSON result outputs for baselines and DQN training/evaluation
- smoke tests for the environment and the main scripts

Planned but not included yet:

- finished analysis notebooks
- checkpoint resume / experiment tracking
- multi-intersection extensions

## Project Goal

The controller chooses whether to keep or switch the traffic-light phase at each step. The objective is to reduce congestion and waiting time while accounting for switching costs.

Core question:

Can an RL policy learn a better long-horizon controller than fixed-cycle and queue-based heuristics, especially when traffic demand changes over time?

## Repository Layout

```text
RL_traffic_Alex/
├── configs/
│   ├── ablations.yaml
│   └── default.yaml
├── docs/
│   └── proposal_draft.md
├── notebooks/
│   └── README.md
├── results/
├── scripts/
│   ├── plot_ablations.py
│   ├── run_ablations.py
│   ├── run_baselines.py
│   ├── summarize_results.py
│   └── train_dqn.py
├── src/
│   └── traffic_rl/
│       ├── __init__.py
│       ├── baselines.py
│       ├── config.py
│       ├── dqn.py
│       ├── env.py
│       └── evaluation.py
├── tests/
│   ├── test_config_and_scripts.py
│   └── test_env.py
├── requirements.txt
└── requirements-optional.txt
```

## Environment Summary

- phase `0`: north-south green
- phase `1`: east-west green
- action `0`: keep current phase
- action `1`: switch phase
- default observation:
  - queue lengths for `N, S, E, W`
  - current phase
  - current phase duration
  - whether a switch is currently allowed
  - remaining yellow time
  - normalized episode step
  - recent average arrivals for `N, S, E, W`

The simulator includes:

- Poisson arrivals from configurable piecewise demand regimes
- per-step departure capacity from the currently green approaches
- minimum-green constraints
- yellow-time switch loss with a pending next phase
- explicit invalid switch request metrics
- configurable `observation_variant` for ablations:
  - `full`: current 13D observation
  - `minimal`: 6D observation with queues, phase, and phase duration
- queue-based or waiting-based reward shaping

## Setup

Core runtime:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On this machine, the verified environment is:

```bash
conda activate clean311
```

Optional extras for notebooks, plotting, or alternative YAML parsing:

```bash
pip install -r requirements-optional.txt
```

`PyYAML` is optional now. The project can read the included config files without it.

## Verification

Run the tests:

```bash
conda run -n clean311 python -m unittest discover -s tests
```

Run baseline evaluation:

```bash
conda run -n clean311 python scripts/run_baselines.py --config configs/default.yaml
conda run -n clean311 python scripts/summarize_results.py results/baseline_summary.json
```

Train and evaluate DQN:

```bash
conda run -n clean311 python scripts/train_dqn.py --config configs/default.yaml
conda run -n clean311 python scripts/summarize_results.py results/dqn_summary.json
```

Run the ablation suite and generate figures:

```bash
conda run -n clean311 python scripts/run_ablations.py --config configs/ablations.yaml
conda run -n clean311 python scripts/plot_ablations.py results/ablations/ablation_summary.json
```

## Outputs

Main generated artifacts:

- `results/baseline_summary.json`
- `results/dqn_summary.json`
- `results/checkpoints/dqn_policy.pt`
- `results/ablations/ablation_summary.json`
- `results/figures/*.png`

Reported metrics:

- `total_reward`
- `average_queue_length`
- `maximum_queue_length`
- `throughput_per_step`
- `total_departed`
- `average_wait_time_steps`
- `average_wait_time_seconds`
- `switch_count`
- `switch_requested_count`
- `switch_applied_count`
- `invalid_switch_count`

## Compatibility Note

Older checkpoints from the previous 10D observation version are not compatible
with this final 13D observation version. Re-run `scripts/train_dqn.py` after
updating the code.

## Known Limitations

- only a single intersection is modeled
- demand is synthetic rather than data-driven
- there is no checkpoint resume path yet
- notebooks are still placeholders
- experiment tracking is minimal and file-based

## Recommended Next Steps

1. Run the baselines and DQN pipeline once end to end.
2. Run `scripts/run_ablations.py` to generate seeded reward/state/switch-penalty studies.
3. Use `scripts/plot_ablations.py` to generate presentation-ready comparison figures.
4. Use notebooks only as a lightweight analysis layer on top of the saved JSON outputs.
