"""Shared experiment helpers for training and aggregating DQN runs."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np

from .env import build_action_mask
from .evaluation import evaluate_policies
from .factory import make_baseline_policies, make_environment, resolve_network_type


def linear_epsilon(global_step: int, start: float, end: float, decay_steps: int) -> float:
    """Linearly anneal epsilon for epsilon-greedy exploration."""
    if global_step >= decay_steps:
        return end
    fraction = global_step / max(decay_steps, 1)
    return start + fraction * (end - start)


def set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def make_dqn_policy(agent: Any, use_action_mask: bool = True):
    """Wrap the agent as an evaluation policy, optionally applying action masks."""

    def policy(observation: np.ndarray, info: Mapping[str, Any] | None = None) -> int:
        action_mask = None
        if use_action_mask:
            action_mask = build_action_mask(
                observation,
                info=info,
                action_dim=agent.action_dim,
            )
        return agent.act(
            observation,
            epsilon=0.0,
            action_mask=action_mask,
        )

    return policy


def train_and_evaluate_dqn(
    config: Mapping[str, Any],
    checkpoint_path: str | Path,
    summary_path: str | Path | None = None,
    run_metadata: Mapping[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train one DQN run, evaluate it against baselines, and persist outputs."""
    from .dqn import DQNAgent, DQNConfig

    env_config = dict(config["environment"])
    train_config = dict(config["training"])
    eval_config = dict(config["evaluation"])
    base_seed = int(train_config.get("seed", 0))

    set_global_seeds(base_seed)

    train_env = make_environment(
        env_config,
        env_config["train_schedule"],
        seed=base_seed,
    )

    gradient_clip_raw = train_config.get("gradient_clip_norm")
    gradient_clip_norm = (
        None if gradient_clip_raw is None else float(gradient_clip_raw)
    )
    double_dqn = bool(train_config.get("double_dqn", True))
    use_action_mask = bool(train_config.get("use_action_mask", True))

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
            double_dqn=double_dqn,
            gradient_clip_norm=gradient_clip_norm,
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
    log_interval = max(int(train_config.get("log_interval_episodes", 25)), 1)
    epsilon = end_epsilon

    if verbose:
        algorithm_name = "Double DQN" if double_dqn else "DQN"
        print(f"Training {algorithm_name}...\n")

    for episode_idx in range(episodes):
        observation, info = train_env.reset(seed=base_seed + episode_idx)
        done = False
        losses = []

        while not done:
            epsilon = linear_epsilon(global_step, start_epsilon, end_epsilon, epsilon_decay_steps)
            action_mask = None
            if use_action_mask:
                action_mask = build_action_mask(
                    observation,
                    info=info,
                    action_dim=agent.action_dim,
                )
            action = agent.act(
                observation,
                epsilon=epsilon,
                action_mask=action_mask,
            )
            next_observation, reward, terminated, truncated, next_info = train_env.step(action)
            done = bool(terminated or truncated)
            next_action_mask = None
            if use_action_mask:
                next_action_mask = build_action_mask(
                    next_observation,
                    info=next_info,
                    action_dim=agent.action_dim,
                )
            agent.observe(
                observation,
                action,
                reward,
                next_observation,
                done,
                next_action_mask=next_action_mask,
            )

            if global_step >= warmup_steps and global_step % update_frequency == 0:
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)

            observation = next_observation
            info = next_info
            global_step += 1

        summary = train_env.summarize()
        summary["episode"] = float(episode_idx)
        summary["epsilon"] = float(epsilon)
        if losses:
            summary["mean_loss"] = float(sum(losses) / len(losses))
        training_history.append(summary)

        if verbose and ((episode_idx + 1) % log_interval == 0 or episode_idx == 0):
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

    policies = make_baseline_policies(train_env)
    policies["dqn"] = make_dqn_policy(agent, use_action_mask=use_action_mask)

    evaluation_results: dict[str, dict[str, dict[str, float]]] = {}
    if verbose:
        print("\nEvaluating trained DQN...\n")

    for regime_name, schedule in env_config["evaluation_regimes"].items():
        env_factory = lambda schedule=schedule: make_environment(env_config, schedule)
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
        "network_type": resolve_network_type(env_config),
        "grid_shape": env_config.get("grid_shape"),
        "intersection_ids": env_config.get("intersection_ids"),
        "seed": base_seed,
        "reward_mode": env_config.get("reward_mode", "queue"),
        "switch_penalty": float(env_config.get("switch_penalty", 2.0)),
        "observation_variant": env_config.get("observation_variant", "full"),
        "episodes": episodes,
        "evaluation_episodes_per_regime": int(eval_config.get("episodes_per_regime", 10)),
        "train_schedule_name": str(env_config.get("train_schedule_name", "train_schedule")),
        "evaluation_regimes": list(env_config["evaluation_regimes"].keys()),
        "double_dqn": double_dqn,
        "use_action_mask": use_action_mask,
        "gradient_clip_norm": gradient_clip_norm,
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


def train_and_evaluate_dqn_multiseed(
    config: Mapping[str, Any],
    seeds: Sequence[int],
    output_dir: str | Path,
    summary_path: str | Path,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train and evaluate DQN over multiple seeds, then aggregate results."""
    seed_values = [int(seed) for seed in seeds]
    if not seed_values:
        raise ValueError("seeds must contain at least one value")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env_config = dict(config["environment"])
    train_config = dict(config["training"])
    episode_length = int(env_config.get("episode_length", 200))

    if verbose:
        print(f"Running multi-seed DQN experiment for seeds={seed_values}\n")

    run_payloads: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    for seed in seed_values:
        run_config = deepcopy(dict(config))
        run_config.setdefault("training", {})
        run_config["training"]["seed"] = seed

        run_dir = output_dir / f"seed_{seed}"
        checkpoint_path = run_dir / "dqn_policy.pt"
        per_seed_summary_path = run_dir / "dqn_summary.json"

        if verbose:
            print(f"Seed {seed}:")

        payload = train_and_evaluate_dqn(
            config=run_config,
            checkpoint_path=checkpoint_path,
            summary_path=per_seed_summary_path,
            run_metadata={
                "seed": seed,
                "multiseed": True,
            },
            verbose=verbose,
        )
        enriched_payload = _with_frequency_metrics(payload, episode_length)
        run_payloads.append(enriched_payload)
        run_records.append(
            {
                "seed": seed,
                "summary_path": str(per_seed_summary_path),
                "checkpoint": str(checkpoint_path),
                "metadata": enriched_payload["metadata"],
                "final_training_episode": enriched_payload["training_history"][-1],
                "evaluation_results": enriched_payload["evaluation_results"],
            }
        )

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_count": len(seed_values),
        "network_type": resolve_network_type(env_config),
        "grid_shape": env_config.get("grid_shape"),
        "intersection_ids": env_config.get("intersection_ids"),
        "train_schedule_name": str(env_config.get("train_schedule_name", "train_schedule")),
        "evaluation_regimes": list(env_config["evaluation_regimes"].keys()),
        "reward_mode": env_config.get("reward_mode", "queue"),
        "switch_penalty": float(env_config.get("switch_penalty", 2.0)),
        "observation_variant": env_config.get("observation_variant", "full"),
        "episodes": int(train_config.get("episodes", 250)),
        "evaluation_episodes_per_regime": int(
            dict(config["evaluation"]).get("episodes_per_regime", 10)
        ),
        "double_dqn": bool(train_config.get("double_dqn", True)),
        "use_action_mask": bool(train_config.get("use_action_mask", True)),
        "gradient_clip_norm": train_config.get("gradient_clip_norm"),
    }
    summary = {
        "metadata": metadata,
        "config": deepcopy(dict(config)),
        "seeds": seed_values,
        "runs": run_records,
        "aggregate": aggregate_run_payloads(run_payloads),
    }

    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if verbose:
        print(f"\nSaved multi-seed summary to {summary_path}")

    return summary


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


def _with_frequency_metrics(payload: Mapping[str, Any], episode_length: int) -> dict[str, Any]:
    enriched = deepcopy(dict(payload))
    if enriched.get("training_history"):
        enriched["training_history"][-1] = _add_frequency_metrics(
            enriched["training_history"][-1],
            episode_length=episode_length,
        )
    for policy_results in enriched.get("evaluation_results", {}).values():
        for policy_name, metrics in list(policy_results.items()):
            policy_results[policy_name] = _add_frequency_metrics(
                metrics,
                episode_length=episode_length,
            )
    return enriched


def _add_frequency_metrics(
    metrics: Mapping[str, float],
    episode_length: int,
) -> dict[str, float]:
    enriched = dict(metrics)
    denominator = float(max(int(episode_length), 1))
    switch_count = float(
        enriched.get("switch_applied_count", enriched.get("switch_count", 0.0))
    )
    enriched["switch_frequency_per_step"] = switch_count / denominator
    enriched["switch_request_frequency_per_step"] = (
        float(enriched.get("switch_requested_count", 0.0)) / denominator
    )
    enriched["invalid_action_frequency_per_step"] = (
        float(enriched.get("invalid_switch_count", 0.0)) / denominator
    )
    return enriched


def _mean_and_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }
