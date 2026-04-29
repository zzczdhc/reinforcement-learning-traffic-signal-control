# Reinforcement Learning for Adaptive Traffic Signal Control

This repository contains a course final project on adaptive traffic signal control at single-intersection (`1x1`) and small grid (`2x2`) networks under stationary and nonstationary demand.

## Status

Implemented now:

- Gymnasium-compatible single-intersection simulator with stochastic arrivals
- Gymnasium-compatible `2x2` grid simulator with centralized joint actions
- minimum-green enforcement, true yellow transitions, and invalid-switch tracking
- three heuristic baselines: fixed-cycle, queue-threshold, max-pressure
- DQN training loop with replay buffer, target network, legal-action masking, Double DQN targets, and seeded runs
- multi-seed DQN experiment aggregation with mean/std reporting
- ablation runner for Double DQN and action masking studies
- CLI config overrides for quick DQN experiments
- lightweight hyperparameter search for both `1x1` and `2x2` setups
- automatic figure generation for DQN runs, ablations, tuning results, and the final clean report assets
- JSON result outputs for baselines, DQN training/evaluation, and tuning
- smoke tests for the environment and the main scripts

## Project Goal

The controller chooses whether to keep or switch the traffic-light phase at each step. The objective is to reduce congestion and waiting time while accounting for switching costs.

Core question:

Can an RL policy learn a better long-horizon controller than fixed-cycle and queue-based heuristics, especially when traffic demand changes over time?

## Repository Layout

```text
reinforcement-learning-traffic-signal-control-final/
├── configs/
│   ├── ablations.yaml
│   ├── default.yaml
│   └── grid_2x2.yaml
├── docs/
│   └── proposal_draft.md
├── notebooks/
│   ├── 01_project_overview.ipynb
│   ├── 02_baseline_comparison.ipynb
│   ├── 03_dqn_training_analysis.ipynb
│   └── README.md
├── results/
├── scripts/
│   ├── build_final_assets.py
│   ├── plot_ablations.py
│   ├── plot_results.py
│   ├── run_ablations.py
│   ├── run_baselines.py
│   ├── summarize_results.py
│   ├── train_dqn.py
│   └── tune_dqn.py
├── src/
│   └── traffic_rl/
│       ├── baselines.py
│       ├── config.py
│       ├── dqn.py
│       ├── env.py
│       ├── evaluation.py
│       ├── experiment.py
│       ├── experiments.py
│       ├── factory.py
│       ├── grid_env.py
│       ├── tuning.py
│       └── visualization.py
├── tests/
│   ├── test_config_and_scripts.py
│   ├── test_config_and_tuning.py
│   ├── test_env.py
│   ├── test_experiment.py
│   └── test_grid_env.py
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
- configurable `observation_variant`:
  - `full`: current 13D observation
  - `minimal`: 6D observation with queues, phase, and phase duration
- queue-based or waiting-based reward shaping

For `2x2`, the centralized DQN chooses one joint action over four intersections. Each intersection still has local actions `0=keep` and `1=switch`; the joint action is the bit encoding of those four local decisions, so the DQN action space has `2^4 = 16` actions. The environment exposes a 16D legal-action mask so combinations that request a switch at an intersection still inside minimum-green/yellow constraints are never selected during masked DQN training or evaluation.

## Setup

Core runtime:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional extras for YAML parsing, plotting, notebooks, and tests:

```bash
pip install -r requirements-optional.txt
```

`PyYAML` and `matplotlib` are optional. The repo can train and evaluate with the core requirements, while plotting features and YAML writing are enabled automatically when the optional packages are installed.

## Verification

Run the tests:

```bash
python3 -m unittest discover -s tests
```

Run baseline evaluation:

```bash
python3 scripts/run_baselines.py --config configs/default.yaml
python3 scripts/summarize_results.py results/baseline_summary.json
```

Train and evaluate DQN:

```bash
python3 scripts/train_dqn.py --config configs/default.yaml
python3 scripts/summarize_results.py results/dqn_summary.json
```

The default `1x1` config uses the tuned non-architecture Double DQN setting from
`configs/default.yaml`: `learning_rate=0.00025`, `batch_size=32`,
`gamma=0.95`, `end_epsilon=0.08`, `epsilon_decay_steps=30000`,
`warmup_steps=800`, `target_sync_steps=50`, `switch_penalty=2.5`, and
`gradient_clip_norm=10.0`.

Run a multi-seed DQN experiment without replacing the single-seed summary:

```bash
python3 scripts/train_dqn.py \
  --config configs/default.yaml \
  --seeds 7,17,27 \
  --no-plots
```

This writes the aggregate to `results/dqn_multiseed_summary.json` and per-seed
checkpoints/summaries under `results/multiseed/`.

Run the standard `1x1` or `2x2` experiment profiles:

```bash
python3 scripts/train_dqn.py --profile 1x1
python3 scripts/train_dqn.py --profile 2x2
```

Try a quick manual parameter override without editing YAML:

```bash
python3 scripts/train_dqn.py \
  --config configs/default.yaml \
  --set training.learning_rate=0.0005 \
  --set training.hidden_dims='[256, 128]'
```

Double DQN is enabled by default in the provided configs. To run the vanilla DQN
target update for an ablation:

```bash
python3 scripts/train_dqn.py \
  --config configs/default.yaml \
  --set training.double_dqn=false
```

Run the built-in hyperparameter search:

```bash
python3 scripts/tune_dqn.py --config configs/default.yaml
python3 scripts/tune_dqn.py --profile 2x2
```

Regenerate plots from an existing summary JSON:

```bash
python3 scripts/plot_results.py \
  --summary results/dqn_summary.json \
  --output-dir results/plots/manual
```

Run the ablation suite and generate figures:

```bash
python3 scripts/run_ablations.py --config configs/ablations.yaml
python3 scripts/build_final_assets.py --results-root results/final_clean
```

The ablation suite compares vanilla DQN vs Double DQN and legal-action masking
on vs off. In separate reward-design experiments, queue-based and waiting-based
reward modes gave very similar results; the report uses queue mode for the final
model because it is simple and directly aligned with congestion reduction.

## Tuning Workflow

- Start with `--profile 1x1` for the compact single-intersection setting, then use `--profile 2x2` with `configs/grid_2x2.yaml` once the basic DQN behavior is stable.
- Put search candidates under `tuning.search_space` in `configs/default.yaml` or `configs/grid_2x2.yaml`.
- Use `tuning.fixed_overrides` to shorten each trial, then re-run the best config with full training episodes.
- By default the `1x1` tuning objective is `dqn` performance on `nonstationary`; the `2x2` objective uses `grid_nonstationary`. Both minimize `average_wait_time_seconds`.

When tuning is run, it writes artifacts such as:

- `results/tuning/tuning_summary.json`
- `results/tuning/1x1/tuning_summary.json`
- `results/tuning/2x2/tuning_summary.json`
- `results/tuning/best_config.yaml`
- `results/plots/tuning/tuning_overview.png`
- `results/plots/tuning/best_trial/*.png`

These transient tuning folders are not part of the clean final submission. The
submitted final statistics and figures live under `results/final_clean/`.

## Outputs

Main generated artifacts:

- `results/final_clean/baselines/baseline_summary.json`
- `results/final_clean/final_double_dqn/dqn_multiseed_summary.json`
- `results/final_clean/original_double_dqn/dqn_multiseed_summary.json`
- `results/final_clean/ablations/ablation_summary.json`
- `results/final_clean/grid_2x2/baselines/baseline_summary.json`
- `results/final_clean/grid_2x2/dqn_multiseed_summary.json`
- `results/final_clean/figures/*.png`
- `results/final_clean/figures/*.pdf`
- `results/final_clean/tables/*.csv`
- `results/final_clean/final_statistics.md`

The 2x2 grid extension is trained with Double DQN and action masking enabled.
Its presentation figures are `fig08_2x2_wait_rl_vs_baselines` and
`fig09_2x2_queue_rl_vs_baselines` under `results/final_clean/figures/`.

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
- `switch_frequency_per_step` in multi-seed aggregates
- `switch_request_frequency_per_step` in multi-seed aggregates
- `invalid_action_frequency_per_step` in multi-seed aggregates
- `internal_transfer_count` for `2x2`
- `average_switches_per_intersection` for `2x2`

## Compatibility Note

Older checkpoints from the previous 10D observation version are not compatible with this final 13D observation version. Re-run `scripts/train_dqn.py` after updating the code.

## Known Limitations

- the `2x2` grid is still a lightweight synthetic simulator, not a calibrated road network
- demand is synthetic rather than data-driven
- there is no checkpoint resume path yet
- experiment tracking is minimal and file-based

## Recommended Next Steps

1. Use `results/final_clean/final_statistics.md` for the final numerical summary.
2. Use `results/final_clean/figures/` for presentation-ready figures.
3. Treat the `2x2` simulator as future work or an extension, not the main reported result.
