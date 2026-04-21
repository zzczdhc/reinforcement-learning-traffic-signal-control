"""Shared experiment helpers for training and aggregating DQN runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .baselines import FixedCycleController, MaxPressureController, QueueThresholdController
from .config import build_env_kwargs
from .dqn import DQNAgent, DQNConfig
from .env import AdaptiveTrafficSignalEnv
from .evaluation import evaluate_policies


def linear_epsilon(global_step: int, start: float, end: float, decay_steps: int) -> float:
    """Linearly anneal epsilon for epsilon-greedy exploration."""
    if global_step >= decay_steps:
        return end
    fraction = global_step / max(decay_steps, 1)
    return start + fraction * (end - start)


def train_and_evaluate_dqn(
    config: Mapping[str, Any],
    checkpoint_path: str | Path,
    summary_path: str | Path | None = None,
    run_metadata: Mapping[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train one DQN run, evaluate it against baselines, and persist outputs."""
    env_config = dict(config["environment"])
    train_config = dict(config["training"])
    eval_config = dict(config["evaluation"])

    train_env = AdaptiveTrafficSignalEnv(
        **build_env_kwargs(env_config, env_config["train_schedule"]),
        seed=int(train_config.get("seed", 0)),
    )

    agent = DQNAgent(
        observation_dim=train_env.observation_dim,
        action_dim=train_env.action_dim,
        config=DQNConfig(
            gamma=float(train_config.get("gamma", 0.99)),
            learning_rate=float(train_config.get("learning_rate", 1e-3)),
            batch_size=int(train_config.get("batch_size", 64)),
            buffer_size=int(train_config.get("buffer_size", 50_000)),
            hidden_dims=tuple(train_config.get("hidden_dims", [128, 128])),
            target_sync_steps=int(train_config.get("target_sync_steps", 250)),
            device=str(train_config.get("device", "cpu")),
        ),
    )

    global_step = 0
    training_history: list[dict[str, float]] = []
    episodes = int(train_config.get("episodes", 250))
    warmup_steps = int(train_config.get("warmup_steps", 500))
    update_frequency = int(train_config.get("update_frequency", 1))
    start_epsilon = float(train_config.get("start_epsilon", 1.0))
    end_epsilon = float(train_config.get("end_epsilon", 0.05))
    epsilon_decay_steps = int(train_config.get("epsilon_decay_steps", 20_000))
    base_seed = int(train_config.get("seed", 0))
    epsilon = end_epsilon

    if verbose:
        print("Training DQN...\n")

    for episode_idx in range(episodes):
        observation, _ = train_env.reset(seed=base_seed + episode_idx)
        done = False
        losses = []

        while not done:
            epsilon = linear_epsilon(global_step, start_epsilon, end_epsilon, epsilon_decay_steps)
            action = agent.act(observation, epsilon=epsilon)
            next_observation, reward, terminated, truncated, _ = train_env.step(action)
            done = bool(terminated or truncated)
            agent.observe(observation, action, reward, next_observation, done)

            if global_step >= warmup_steps and global_step % update_frequency == 0:
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)

            observation = next_observation
            global_step += 1

        summary = train_env.summarize()
        summary["episode"] = float(episode_idx)
        summary["epsilon"] = float(epsilon)
        if losses:
            summary["mean_loss"] = float(sum(losses) / len(losses))
        training_history.append(summary)

        if verbose and ((episode_idx + 1) % 25 == 0 or episode_idx == 0):
            print(
                f"Episode {episode_idx + 1:4d}/{episodes} | "
                f"reward={summary['total_reward']:.2f} | "
                f"avg_queue={summary['average_queue_length']:.2f} | "
                f"avg_wait_s={summary['average_wait_time_seconds']:.2f} | "
                f"invalid={summary['invalid_switch_count']:.0f} | "
                f"epsilon={summary['epsilon']:.3f}"
            )

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(checkpoint_path))

    policies = {
        "fixed_cycle": FixedCycleController(cycle_length=10),
        "queue_threshold": QueueThresholdController(threshold=5.0, min_green=3),
        "max_pressure": MaxPressureController(min_green=2),
        "dqn": lambda observation, info=None: agent.act(observation, epsilon=0.0),
    }

    evaluation_results: dict[str, dict[str, dict[str, float]]] = {}
    if verbose:
        print("\nEvaluating trained DQN...\n")

    for regime_name, schedule in env_config["evaluation_regimes"].items():
        env_kwargs = build_env_kwargs(env_config, schedule)
        env_factory = lambda env_kwargs=env_kwargs: AdaptiveTrafficSignalEnv(**env_kwargs)
        regime_results = evaluate_policies(
            env_factory=env_factory,
            policies=policies,
            episodes=int(eval_config.get("episodes_per_regime", 10)),
            base_seed=10_000,
        )
        evaluation_results[regime_name] = regime_results

        if verbose:
            dqn_summary = regime_results["dqn"]
            print(
                f"{regime_name:18s} | "
                f"avg_queue={dqn_summary['average_queue_length']:.2f} | "
                f"avg_wait_s={dqn_summary['average_wait_time_seconds']:.2f} | "
                f"throughput={dqn_summary['throughput_per_step']:.2f} | "
                f"invalid={dqn_summary['invalid_switch_count']:.2f}"
            )

    metadata = {
        "study_name": "default",
        "variant_name": "default",
        "seed": base_seed,
        "reward_mode": env_config.get("reward_mode", "queue"),
        "switch_penalty": float(env_config.get("switch_penalty", 2.0)),
        "observation_variant": env_config.get("observation_variant", "full"),
        "episodes": episodes,
        "evaluation_episodes_per_regime": int(eval_config.get("episodes_per_regime", 10)),
        "train_schedule_name": "train_schedule",
        "evaluation_regimes": list(env_config["evaluation_regimes"].keys()),
    }
    if run_metadata:
        metadata.update(run_metadata)

    payload: dict[str, Any] = {
        "metadata": metadata,
        "training_history": training_history,
        "evaluation_results": evaluation_results,
        "checkpoint": str(checkpoint_path),
    }

    if summary_path is not None:
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if verbose:
        print(f"\nSaved checkpoint to {checkpoint_path}")
        if summary_path is not None:
            print(f"Saved training summary to {summary_path}")

    return payload


def aggregate_run_payloads(run_payloads: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate multiple DQN payloads into mean/std summaries."""
    if not run_payloads:
        raise ValueError("run_payloads must contain at least one payload")

    aggregated_regimes: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    example_results = run_payloads[0]["evaluation_results"]

    for regime_name, policy_results in example_results.items():
        aggregated_regimes[regime_name] = {}
        for policy_name, metrics in policy_results.items():
            aggregated_metrics: dict[str, dict[str, float]] = {}
            for metric_name in metrics:
                values = [
                    float(payload["evaluation_results"][regime_name][policy_name][metric_name])
                    for payload in run_payloads
                ]
                aggregated_metrics[metric_name] = _mean_and_std(values)
            aggregated_regimes[regime_name][policy_name] = aggregated_metrics

    final_training_episode = [
        payload["training_history"][-1]
        for payload in run_payloads
        if payload.get("training_history")
    ]
    aggregated_training: dict[str, dict[str, float]] = {}
    if final_training_episode:
        for metric_name in final_training_episode[0]:
            values = [float(summary[metric_name]) for summary in final_training_episode]
            aggregated_training[metric_name] = _mean_and_std(values)

    return {
        "run_count": len(run_payloads),
        "per_regime": aggregated_regimes,
        "final_training_episode": aggregated_training,
    }


def _mean_and_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }
