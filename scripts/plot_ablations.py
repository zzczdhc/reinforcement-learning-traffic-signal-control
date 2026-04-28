#!/usr/bin/env python3
"""Generate presentation-ready plots from ablation summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POLICY_ORDER = ["fixed_cycle", "queue_threshold", "max_pressure", "dqn"]
POLICY_COLORS = {
    "fixed_cycle": "#4e79a7",
    "queue_threshold": "#59a14f",
    "max_pressure": "#f28e2b",
    "dqn": "#e15759",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_grouped_metric(
    aggregate: dict,
    metric_name: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    regimes = list(aggregate.keys())
    positions = np.arange(len(regimes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for idx, policy_name in enumerate(POLICY_ORDER):
        means = [aggregate[regime][policy_name][metric_name]["mean"] for regime in regimes]
        stds = [aggregate[regime][policy_name][metric_name]["std"] for regime in regimes]
        offset = (idx - (len(POLICY_ORDER) - 1) / 2.0) * width
        ax.bar(
            positions + offset,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            label=policy_name,
            color=POLICY_COLORS[policy_name],
            alpha=0.9,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(regimes, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_variant_metric(
    summary: dict,
    study_name: str,
    metric_name: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    study = summary["studies"].get(study_name)
    if not study:
        return

    variant_names = study["variant_order"]
    means = []
    stds = []
    for variant_name in variant_names:
        seed_scores = []
        for run in study["variants"][variant_name]["runs"]:
            payload = load_json(Path(run["summary_path"]))
            regime_scores = [
                float(policy_results["dqn"][metric_name])
                for policy_results in payload["evaluation_results"].values()
            ]
            seed_scores.append(float(np.mean(regime_scores)))
        means.append(float(np.mean(seed_scores)))
        stds.append(float(np.std(seed_scores)))

    positions = np.arange(len(variant_names))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        positions,
        means,
        yerr=stds,
        capsize=5,
        color="#9c755f",
        alpha=0.9,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(variant_names, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def find_reference_variant(summary: dict) -> tuple[str, str]:
    preferred_variants = [
        ("algorithm_update", "double_dqn"),
    ]
    for study_name, variant_name in preferred_variants:
        study = summary["studies"].get(study_name)
        if study and variant_name in study["variants"]:
            return study_name, variant_name

    for study_name, study in summary["studies"].items():
        if study["variant_order"]:
            return study_name, study["variant_order"][0]
    raise ValueError("ablation summary does not contain any variants")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=str, help="Path to ablation_summary.json")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write plots to. Defaults to results/figures next to the summary root.",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary).resolve()
    summary = load_json(summary_path)

    if args.output_dir is None:
        output_dir = summary_path.parent.parent / "figures"
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    study_name, variant_name = find_reference_variant(summary)
    reference_variant = summary["studies"][study_name]["variants"][variant_name]
    reference_aggregate = reference_variant["aggregate"]["per_regime"]

    plot_grouped_metric(
        aggregate=reference_aggregate,
        metric_name="average_queue_length",
        title=f"Baselines vs DQN: Average Queue ({study_name}/{variant_name})",
        ylabel="Average Queue Length",
        output_path=output_dir / "baseline_vs_dqn_avg_queue.png",
    )
    plot_grouped_metric(
        aggregate=reference_aggregate,
        metric_name="average_wait_time_seconds",
        title=f"Baselines vs DQN: Average Wait ({study_name}/{variant_name})",
        ylabel="Average Wait Time (seconds)",
        output_path=output_dir / "baseline_vs_dqn_avg_wait.png",
    )

    plot_variant_metric(
        summary=summary,
        study_name="algorithm_update",
        metric_name="average_wait_time_seconds",
        title="DQN Ablation: Mean Wait Across Regimes",
        ylabel="Average Wait Time (seconds)",
        output_path=output_dir / "double_dqn_ablation_avg_wait.png",
    )
    plot_variant_metric(
        summary=summary,
        study_name="algorithm_update",
        metric_name="average_queue_length",
        title="DQN Ablation: Mean Queue Across Regimes",
        ylabel="Average Queue Length",
        output_path=output_dir / "double_dqn_ablation_avg_queue.png",
    )
    plot_variant_metric(
        summary=summary,
        study_name="action_masking",
        metric_name="invalid_switch_count",
        title="Action Mask Ablation: Invalid Switch Requests",
        ylabel="Invalid Switch Count",
        output_path=output_dir / "action_mask_invalid_switch.png",
    )
    plot_variant_metric(
        summary=summary,
        study_name="action_masking",
        metric_name="average_wait_time_seconds",
        title="Action Mask Ablation: Mean Wait Across Regimes",
        ylabel="Average Wait Time (seconds)",
        output_path=output_dir / "action_mask_avg_wait.png",
    )

    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
