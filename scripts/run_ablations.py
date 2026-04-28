#!/usr/bin/env python3
"""Run ablation studies over multiple seeded DQN experiments."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_rl.config import load_config
from traffic_rl.experiments import aggregate_run_payloads, train_and_evaluate_dqn


def deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge config overrides into a copy of a base mapping."""
    merged = deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_project_path(path_str: str) -> Path:
    """Resolve a config path relative to the project root when needed."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ablations.yaml",
        help="Path to the ablation study config.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Override the output_root configured in the ablation YAML.",
    )
    args = parser.parse_args()

    ablation_config_path = resolve_project_path(args.config)
    ablation_config = load_config(ablation_config_path)
    base_config_path = resolve_project_path(ablation_config["base_config"])
    base_config = load_config(base_config_path)

    global_defaults = dict(ablation_config.get("global_defaults", {}))
    default_seeds = [int(seed) for seed in global_defaults.get("seeds", [7, 17, 27])]
    output_root_setting = args.output_root or global_defaults.get("output_root", "results/ablations")
    output_root = resolve_project_path(output_root_setting)
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "base_config": str(base_config_path),
        "output_root": str(output_root),
        "studies": {},
    }

    for study_name, study_config in ablation_config["studies"].items():
        print(f"Study: {study_name}")
        variants = study_config["variants"]
        study_summary: dict[str, Any] = {
            "description": study_config.get("description", ""),
            "variant_order": list(variants.keys()),
            "variants": {},
        }

        for variant_name, variant_config in variants.items():
            overrides = dict(variant_config.get("overrides", {}))
            seeds = [int(seed) for seed in variant_config.get("seeds", default_seeds)]
            run_payloads = []
            run_records = []

            print(f"  Variant: {variant_name}")
            for seed in seeds:
                run_config = deep_merge(base_config, overrides)
                run_config.setdefault("training", {})
                run_config["training"]["seed"] = seed

                run_dir = output_root / study_name / variant_name / f"seed_{seed}"
                summary_path = run_dir / "dqn_summary.json"
                checkpoint_path = run_dir / "dqn_policy.pt"

                payload = train_and_evaluate_dqn(
                    config=run_config,
                    checkpoint_path=checkpoint_path,
                    summary_path=summary_path,
                    run_metadata={
                        "study_name": study_name,
                        "variant_name": variant_name,
                        "seed": seed,
                    },
                    verbose=False,
                )
                run_payloads.append(payload)
                run_records.append(
                    {
                        "seed": seed,
                        "summary_path": str(summary_path),
                        "checkpoint": str(checkpoint_path),
                        "metadata": payload["metadata"],
                    }
                )
                print(
                    "    "
                    f"seed={seed} saved={summary_path.name} "
                    f"train={payload['metadata']['train_schedule_name']} "
                    f"obs={payload['metadata']['observation_variant']} "
                    f"reward={payload['metadata']['reward_mode']} "
                    f"switch_penalty={payload['metadata']['switch_penalty']} "
                    f"double_dqn={payload['metadata'].get('double_dqn')} "
                    f"action_mask={payload['metadata'].get('use_action_mask')}"
                )

            study_summary["variants"][variant_name] = {
                "description": variant_config.get("description", ""),
                "overrides": overrides,
                "seeds": seeds,
                "runs": run_records,
                "aggregate": aggregate_run_payloads(run_payloads),
            }
            print()

        summary["studies"][study_name] = study_summary

    summary_path = output_root / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved ablation summary to {summary_path}")


if __name__ == "__main__":
    main()
