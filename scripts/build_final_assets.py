#!/usr/bin/env python3
"""Build clean final figures and statistics for the presentation/report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]

POLICY_ORDER = ["fixed_cycle", "queue_threshold", "max_pressure", "dqn"]
POLICY_LABELS = {
    "fixed_cycle": "Fixed cycle",
    "queue_threshold": "Queue threshold",
    "max_pressure": "Max pressure",
    "dqn": "Final RL agent",
}
REGIME_LABELS = {
    "symmetric_low": "Symmetric low",
    "symmetric_high": "Symmetric high",
    "asymmetric": "Asymmetric",
    "nonstationary": "Nonstationary",
    "burst_east_west": "EW burst",
    "grid_balanced_low": "Grid low",
    "grid_balanced_high": "Grid high",
    "east_west_commute": "EW commute",
    "north_south_commute": "NS commute",
    "grid_nonstationary": "Grid nonstationary",
}
COLORS = {
    "fixed_cycle": "#4C78A8",
    "queue_threshold": "#54A24B",
    "max_pressure": "#F58518",
    "dqn": "#E45756",
    "original": "#4C78A8",
    "tuned": "#E45756",
    "vanilla_dqn": "#4C78A8",
    "double_dqn": "#E45756",
    "mask_off": "#4C78A8",
    "mask_on": "#E45756",
    "wait": "#E45756",
    "queue": "#4C78A8",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def regime_label(regime: str) -> str:
    return REGIME_LABELS.get(regime, regime.replace("_", " ").title())


def metric_mean_std(metric_payload: Any) -> tuple[float, float]:
    if isinstance(metric_payload, Mapping):
        return float(metric_payload["mean"]), float(metric_payload.get("std", 0.0))
    return float(metric_payload), 0.0


def aggregate_regimes(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    return summary["aggregate"]["per_regime"]


def policy_metric(
    summary: Mapping[str, Any],
    regime: str,
    policy: str,
    metric: str,
) -> tuple[float, float]:
    return metric_mean_std(aggregate_regimes(summary)[regime][policy][metric])


def dqn_metric(
    summary: Mapping[str, Any],
    regime: str,
    metric: str,
) -> tuple[float, float]:
    return policy_metric(summary, regime, "dqn", metric)


def study_metric(
    ablation_summary: Mapping[str, Any],
    study_name: str,
    variant_name: str,
    regime: str,
    metric: str,
) -> tuple[float, float]:
    payload = (
        ablation_summary["studies"][study_name]["variants"][variant_name]["aggregate"][
            "per_regime"
        ][regime]["dqn"][metric]
    )
    return metric_mean_std(payload)


def mean_across_regimes(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def percent_reduction(before: float, after: float) -> float:
    if before == 0.0:
        return 0.0
    return (before - after) / before * 100.0


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.8,
        }
    )


def save_figure(fig: plt.Figure, output_base: Path) -> list[str]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for suffix in [".png", ".pdf"]:
        path = output_base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight")
        saved_paths.append(str(path))
    plt.close(fig)
    return saved_paths


def plot_policy_dot(
    summary: Mapping[str, Any],
    metric: str,
    title: str,
    xlabel: str,
    output_base: Path,
) -> list[str]:
    regimes = list(aggregate_regimes(summary).keys())
    y_base = np.arange(len(regimes), dtype=np.float64)
    offsets = np.linspace(-0.24, 0.24, len(POLICY_ORDER))

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for offset, policy in zip(offsets, POLICY_ORDER):
        values = [policy_metric(summary, regime, policy, metric)[0] for regime in regimes]
        ax.scatter(
            values,
            y_base + offset,
            s=90,
            color=COLORS[policy],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )

    ax.set_yticks(y_base)
    ax.set_yticklabels([regime_label(regime) for regime in regimes])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=16)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS[policy],
            markeredgecolor="white",
            markersize=10,
            label=POLICY_LABELS[policy],
        )
        for policy in POLICY_ORDER
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        frameon=False,
        borderaxespad=0.0,
        title="Policy",
    )
    ax.set_xlim(left=0)
    fig.subplots_adjust(right=0.78)
    return save_figure(fig, output_base)


def plot_best_baseline_improvement(
    summary: Mapping[str, Any],
    output_base: Path,
) -> list[str]:
    regimes = list(aggregate_regimes(summary).keys())
    y_base = np.arange(len(regimes), dtype=np.float64)
    wait_improvements = []
    queue_improvements = []

    for regime in regimes:
        dqn_wait = policy_metric(summary, regime, "dqn", "average_wait_time_seconds")[0]
        dqn_queue = policy_metric(summary, regime, "dqn", "average_queue_length")[0]
        baseline_wait = min(
            policy_metric(summary, regime, policy, "average_wait_time_seconds")[0]
            for policy in POLICY_ORDER
            if policy != "dqn"
        )
        baseline_queue = min(
            policy_metric(summary, regime, policy, "average_queue_length")[0]
            for policy in POLICY_ORDER
            if policy != "dqn"
        )
        wait_improvements.append(percent_reduction(baseline_wait, dqn_wait))
        queue_improvements.append(percent_reduction(baseline_queue, dqn_queue))

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for values, offset, label, color in [
        (wait_improvements, -0.12, "Average wait", COLORS["wait"]),
        (queue_improvements, 0.12, "Average queue", COLORS["queue"]),
    ]:
        ax.scatter(values, y_base + offset, s=95, label=label, color=color, zorder=3)
        for value, y_pos in zip(values, y_base + offset):
            ax.plot([0, value], [y_pos, y_pos], color=color, alpha=0.28, linewidth=3)

    ax.axvline(0.0, color="#333333", linewidth=1)
    ax.set_yticks(y_base)
    ax.set_yticklabels([regime_label(regime) for regime in regimes])
    ax.invert_yaxis()
    ax.set_xlabel("Improvement over best baseline (%)")
    ax.set_title("Final RL Agent Improvement Over Best Baseline", pad=16)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        frameon=False,
        title="Metric",
    )
    fig.subplots_adjust(right=0.78)
    return save_figure(fig, output_base)


def plot_variant_comparison(
    regimes: list[str],
    first_values: Mapping[str, list[float]],
    second_values: Mapping[str, list[float]],
    first_label: str,
    second_label: str,
    title: str,
    output_base: Path,
) -> list[str]:
    x = np.arange(len(regimes))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    metric_specs = [
        ("average_wait_time_seconds", "Average wait (s)"),
        ("average_queue_length", "Average queue length"),
    ]
    for ax, (metric, ylabel) in zip(axes, metric_specs):
        ax.plot(
            x,
            first_values[metric],
            marker="o",
            linewidth=2.4,
            markersize=7,
            color=COLORS["original"],
            label=first_label,
        )
        ax.plot(
            x,
            second_values[metric],
            marker="o",
            linewidth=2.4,
            markersize=7,
            color=COLORS["tuned"],
            label=second_label,
        )
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([regime_label(regime) for regime in regimes], rotation=20, ha="right")
        ax.margins(x=0.04)
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.05, 1.02), ncol=2, frameon=False)
    fig.suptitle(title, y=1.04, fontsize=17)
    fig.tight_layout()
    return save_figure(fig, output_base)


def plot_action_mask_comparison(
    ablation_summary: Mapping[str, Any],
    output_base: Path,
) -> list[str]:
    regimes = list(
        ablation_summary["studies"]["action_masking"]["variants"]["mask_on"]["aggregate"][
            "per_regime"
        ].keys()
    )
    x = np.arange(len(regimes))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    specs = [
        ("average_wait_time_seconds", "Average wait (s)"),
        ("invalid_switch_count", "Invalid switch requests"),
    ]
    for ax, (metric, ylabel) in zip(axes, specs):
        off_values = [
            study_metric(ablation_summary, "action_masking", "mask_off", regime, metric)[0]
            for regime in regimes
        ]
        on_values = [
            study_metric(ablation_summary, "action_masking", "mask_on", regime, metric)[0]
            for regime in regimes
        ]
        ax.plot(
            x,
            off_values,
            marker="o",
            linewidth=2.4,
            markersize=7,
            color=COLORS["mask_off"],
            label="Mask off",
        )
        ax.plot(
            x,
            on_values,
            marker="o",
            linewidth=2.4,
            markersize=7,
            color=COLORS["mask_on"],
            label="Mask on",
        )
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([regime_label(regime) for regime in regimes], rotation=20, ha="right")
        ax.margins(x=0.04)
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.05, 1.02), ncol=2, frameon=False)
    fig.suptitle("Action Masking Ablation", y=1.04, fontsize=17)
    fig.tight_layout()
    return save_figure(fig, output_base)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float64)
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        output[idx] = float(np.mean(values[start : idx + 1]))
    return output


def load_training_histories(summary: Mapping[str, Any]) -> list[list[Mapping[str, Any]]]:
    histories = []
    for run in summary.get("runs", []):
        summary_path = Path(run["summary_path"])
        if not summary_path.is_absolute():
            summary_path = PROJECT_ROOT / summary_path
        if not summary_path.exists() and "results" in summary_path.parts:
            results_index = summary_path.parts.index("results")
            summary_path = PROJECT_ROOT.joinpath(*summary_path.parts[results_index:])
        histories.append(load_json(summary_path)["training_history"])
    return histories


def plot_training_curves(
    final_summary: Mapping[str, Any],
    output_base: Path,
    window: int = 10,
) -> list[str]:
    histories = load_training_histories(final_summary)
    if not histories:
        return []
    min_len = min(len(history) for history in histories)
    episodes = np.arange(min_len) + 1
    metrics = [
        ("total_reward", "Episode reward"),
        ("average_queue_length", "Average queue"),
        ("average_wait_time_seconds", "Average wait (s)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for ax, (metric, ylabel) in zip(axes, metrics):
        series = []
        for history in histories:
            raw = np.asarray([float(item[metric]) for item in history[:min_len]], dtype=np.float64)
            series.append(rolling_mean(raw, window=window))
        stacked = np.vstack(series)
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        ax.plot(episodes, mean, color=COLORS["dqn"], linewidth=2.2)
        ax.fill_between(episodes, mean - std, mean + std, color=COLORS["dqn"], alpha=0.18)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Episode")
    fig.suptitle(f"Final Double DQN Training Curves ({window}-episode rolling mean)", y=1.04)
    fig.tight_layout()
    return save_figure(fig, output_base)


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_baseline_rows(final_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for regime in aggregate_regimes(final_summary):
        dqn_wait = policy_metric(final_summary, regime, "dqn", "average_wait_time_seconds")[0]
        dqn_queue = policy_metric(final_summary, regime, "dqn", "average_queue_length")[0]
        best_wait_policy = min(
            (policy for policy in POLICY_ORDER if policy != "dqn"),
            key=lambda policy: policy_metric(final_summary, regime, policy, "average_wait_time_seconds")[0],
        )
        best_queue_policy = min(
            (policy for policy in POLICY_ORDER if policy != "dqn"),
            key=lambda policy: policy_metric(final_summary, regime, policy, "average_queue_length")[0],
        )
        best_wait = policy_metric(final_summary, regime, best_wait_policy, "average_wait_time_seconds")[0]
        best_queue = policy_metric(final_summary, regime, best_queue_policy, "average_queue_length")[0]
        row = {
            "regime": regime,
            "best_wait_baseline": best_wait_policy,
            "best_wait_baseline_value": round(best_wait, 6),
            "rl_wait": round(dqn_wait, 6),
            "rl_wait_improvement_pct": round(percent_reduction(best_wait, dqn_wait), 4),
            "best_queue_baseline": best_queue_policy,
            "best_queue_baseline_value": round(best_queue, 6),
            "rl_queue": round(dqn_queue, 6),
            "rl_queue_improvement_pct": round(percent_reduction(best_queue, dqn_queue), 4),
        }
        for policy in POLICY_ORDER:
            row[f"{policy}_wait"] = round(
                policy_metric(final_summary, regime, policy, "average_wait_time_seconds")[0], 6
            )
            row[f"{policy}_queue"] = round(
                policy_metric(final_summary, regime, policy, "average_queue_length")[0], 6
            )
            row[f"{policy}_throughput"] = round(
                policy_metric(final_summary, regime, policy, "throughput_per_step")[0], 6
            )
            row[f"{policy}_switch_count"] = round(
                policy_metric(final_summary, regime, policy, "switch_count")[0], 6
            )
            row[f"{policy}_invalid_switch_count"] = round(
                policy_metric(final_summary, regime, policy, "invalid_switch_count")[0], 6
            )
        rows.append(row)
    return rows


def collect_pair_rows(
    regimes: list[str],
    first_name: str,
    second_name: str,
    first_getter,
    second_getter,
) -> list[dict[str, Any]]:
    rows = []
    for regime in regimes:
        first_wait = first_getter(regime, "average_wait_time_seconds")
        second_wait = second_getter(regime, "average_wait_time_seconds")
        first_queue = first_getter(regime, "average_queue_length")
        second_queue = second_getter(regime, "average_queue_length")
        rows.append(
            {
                "regime": regime,
                "first_variant": first_name,
                "second_variant": second_name,
                "first_wait": round(first_wait, 6),
                "second_wait": round(second_wait, 6),
                "wait_improvement_pct": round(percent_reduction(first_wait, second_wait), 4),
                "first_queue": round(first_queue, 6),
                "second_queue": round(second_queue, 6),
                "queue_improvement_pct": round(percent_reduction(first_queue, second_queue), 4),
            }
        )
    return rows


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def summarize_results(
    final_summary: Mapping[str, Any],
    original_summary: Mapping[str, Any],
    ablation_summary: Mapping[str, Any],
    grid_summary: Mapping[str, Any] | None,
    tables_dir: Path,
    output_json: Path,
    output_markdown: Path,
) -> None:
    regimes = list(aggregate_regimes(final_summary).keys())
    baseline_rows = collect_baseline_rows(final_summary)
    write_csv(tables_dir / "baseline_vs_rl_by_regime.csv", baseline_rows)

    original_rows = collect_pair_rows(
        regimes,
        "Original Double DQN",
        "Tuned Double DQN",
        lambda regime, metric: dqn_metric(original_summary, regime, metric)[0],
        lambda regime, metric: dqn_metric(final_summary, regime, metric)[0],
    )
    write_csv(tables_dir / "original_vs_tuned_double_dqn.csv", original_rows)

    algorithm_rows = collect_pair_rows(
        regimes,
        "Vanilla DQN",
        "Double DQN",
        lambda regime, metric: study_metric(ablation_summary, "algorithm_update", "vanilla_dqn", regime, metric)[0],
        lambda regime, metric: study_metric(ablation_summary, "algorithm_update", "double_dqn", regime, metric)[0],
    )
    write_csv(tables_dir / "double_dqn_ablation.csv", algorithm_rows)

    action_rows = collect_pair_rows(
        regimes,
        "Mask off",
        "Mask on",
        lambda regime, metric: study_metric(ablation_summary, "action_masking", "mask_off", regime, metric)[0],
        lambda regime, metric: study_metric(ablation_summary, "action_masking", "mask_on", regime, metric)[0],
    )
    for row in action_rows:
        regime = row["regime"]
        row["mask_off_invalid"] = round(
            study_metric(ablation_summary, "action_masking", "mask_off", regime, "invalid_switch_count")[0],
            6,
        )
        row["mask_on_invalid"] = round(
            study_metric(ablation_summary, "action_masking", "mask_on", regime, "invalid_switch_count")[0],
            6,
        )
    write_csv(tables_dir / "action_mask_ablation.csv", action_rows)

    grid_rows: list[dict[str, Any]] = []
    grid_overall: dict[str, float] | None = None
    if grid_summary is not None:
        grid_rows = collect_baseline_rows(grid_summary)
        write_csv(tables_dir / "grid_2x2_baseline_vs_rl_by_regime.csv", grid_rows)
        grid_regimes = list(aggregate_regimes(grid_summary).keys())
        grid_overall = {
            "mean_wait_improvement_pct": mean_across_regimes(
                row["rl_wait_improvement_pct"] for row in grid_rows
            ),
            "mean_queue_improvement_pct": mean_across_regimes(
                row["rl_queue_improvement_pct"] for row in grid_rows
            ),
            "average_wait_time_seconds": mean_across_regimes(
                policy_metric(grid_summary, regime, "dqn", "average_wait_time_seconds")[0]
                for regime in grid_regimes
            ),
            "average_queue_length": mean_across_regimes(
                policy_metric(grid_summary, regime, "dqn", "average_queue_length")[0]
                for regime in grid_regimes
            ),
            "throughput_per_step": mean_across_regimes(
                policy_metric(grid_summary, regime, "dqn", "throughput_per_step")[0]
                for regime in grid_regimes
            ),
            "invalid_switch_count": mean_across_regimes(
                policy_metric(grid_summary, regime, "dqn", "invalid_switch_count")[0]
                for regime in grid_regimes
            ),
        }

    final_rl_overall = {
        "average_wait_time_seconds": mean_across_regimes(
            policy_metric(final_summary, regime, "dqn", "average_wait_time_seconds")[0]
            for regime in regimes
        ),
        "average_queue_length": mean_across_regimes(
            policy_metric(final_summary, regime, "dqn", "average_queue_length")[0]
            for regime in regimes
        ),
        "throughput_per_step": mean_across_regimes(
            policy_metric(final_summary, regime, "dqn", "throughput_per_step")[0]
            for regime in regimes
        ),
        "switch_count": mean_across_regimes(
            policy_metric(final_summary, regime, "dqn", "switch_count")[0]
            for regime in regimes
        ),
        "invalid_switch_count": mean_across_regimes(
            policy_metric(final_summary, regime, "dqn", "invalid_switch_count")[0]
            for regime in regimes
        ),
    }

    overall = {
        "final_rl_agent": final_rl_overall,
        "rl_vs_best_baseline": {
            "mean_wait_improvement_pct": mean_across_regimes(
                row["rl_wait_improvement_pct"] for row in baseline_rows
            ),
            "mean_queue_improvement_pct": mean_across_regimes(
                row["rl_queue_improvement_pct"] for row in baseline_rows
            ),
        },
        "original_vs_tuned_double_dqn": {
            "mean_wait_improvement_pct": mean_across_regimes(
                row["wait_improvement_pct"] for row in original_rows
            ),
            "mean_queue_improvement_pct": mean_across_regimes(
                row["queue_improvement_pct"] for row in original_rows
            ),
        },
        "double_dqn_ablation": {
            "mean_wait_improvement_pct": mean_across_regimes(
                row["wait_improvement_pct"] for row in algorithm_rows
            ),
            "mean_queue_improvement_pct": mean_across_regimes(
                row["queue_improvement_pct"] for row in algorithm_rows
            ),
        },
        "action_mask_ablation": {
            "mean_wait_improvement_pct": mean_across_regimes(
                row["wait_improvement_pct"] for row in action_rows
            ),
            "mean_queue_improvement_pct": mean_across_regimes(
                row["queue_improvement_pct"] for row in action_rows
            ),
            "mean_invalid_switch_reduction_pct": percent_reduction(
                mean_across_regimes(row["mask_off_invalid"] for row in action_rows),
                mean_across_regimes(row["mask_on_invalid"] for row in action_rows),
            ),
        },
    }
    if grid_overall is not None:
        overall["grid_2x2_extension"] = grid_overall

    write_json(output_json, {"overall": overall, "by_regime": {
        "baseline_vs_rl": baseline_rows,
        "original_vs_tuned_double_dqn": original_rows,
        "double_dqn_ablation": algorithm_rows,
        "action_mask_ablation": action_rows,
        "grid_2x2_baseline_vs_rl": grid_rows,
    }})

    markdown_parts = [
        "# Final Clean Run Statistics",
        "",
        "## Overall Results",
        markdown_table(
            ["Comparison", "Wait improvement", "Queue improvement", "Other"],
            [
                [
                    "Final RL vs best baseline",
                    f"{overall['rl_vs_best_baseline']['mean_wait_improvement_pct']:.2f}%",
                    f"{overall['rl_vs_best_baseline']['mean_queue_improvement_pct']:.2f}%",
                    "Averaged across regimes",
                ],
                [
                    "Tuned vs original Double DQN",
                    f"{overall['original_vs_tuned_double_dqn']['mean_wait_improvement_pct']:.2f}%",
                    f"{overall['original_vs_tuned_double_dqn']['mean_queue_improvement_pct']:.2f}%",
                    "No new hyperparameter search",
                ],
                [
                    "Double DQN vs vanilla DQN",
                    f"{overall['double_dqn_ablation']['mean_wait_improvement_pct']:.2f}%",
                    f"{overall['double_dqn_ablation']['mean_queue_improvement_pct']:.2f}%",
                    "Algorithm ablation",
                ],
                [
                    "Action mask on vs off",
                    f"{overall['action_mask_ablation']['mean_wait_improvement_pct']:.2f}%",
                    f"{overall['action_mask_ablation']['mean_queue_improvement_pct']:.2f}%",
                    f"Invalid switches reduced by {overall['action_mask_ablation']['mean_invalid_switch_reduction_pct']:.2f}%",
                ],
            ]
            + (
                [
                    [
                        "2x2 Double DQN vs best baseline",
                        f"{grid_overall['mean_wait_improvement_pct']:.2f}%",
                        f"{grid_overall['mean_queue_improvement_pct']:.2f}%",
                        "Extension experiment",
                    ]
                ]
                if grid_overall is not None
                else []
            ),
        ),
        "",
        "## Final RL Agent Overall Metrics",
        markdown_table(
            ["Metric", "Mean across regimes"],
            [
                ["Average wait (s)", f"{final_rl_overall['average_wait_time_seconds']:.4f}"],
                ["Average queue length", f"{final_rl_overall['average_queue_length']:.4f}"],
                ["Throughput per step", f"{final_rl_overall['throughput_per_step']:.4f}"],
                ["Switch count", f"{final_rl_overall['switch_count']:.2f}"],
                ["Invalid switch count", f"{final_rl_overall['invalid_switch_count']:.2f}"],
            ],
        ),
        "",
        "## RL vs Best Baseline by Regime",
        markdown_table(
            ["Regime", "Best wait baseline", "Wait improvement", "Best queue baseline", "Queue improvement"],
            [
                [
                    row["regime"],
                    row["best_wait_baseline"],
                    f"{row['rl_wait_improvement_pct']:.2f}%",
                    row["best_queue_baseline"],
                    f"{row['rl_queue_improvement_pct']:.2f}%",
                ]
                for row in baseline_rows
            ],
        ),
    ]
    if grid_overall is not None:
        markdown_parts.extend(
            [
                "",
                "## 2x2 Grid Extension",
                markdown_table(
                    ["Metric", "Mean across grid regimes"],
                    [
                        ["Average wait (s)", f"{grid_overall['average_wait_time_seconds']:.4f}"],
                        ["Average queue length", f"{grid_overall['average_queue_length']:.4f}"],
                        ["Throughput per step", f"{grid_overall['throughput_per_step']:.4f}"],
                        ["Invalid switch count", f"{grid_overall['invalid_switch_count']:.2f}"],
                    ],
                ),
                "",
                markdown_table(
                    [
                        "Regime",
                        "Best wait baseline",
                        "Wait improvement",
                        "Best queue baseline",
                        "Queue improvement",
                    ],
                    [
                        [
                            row["regime"],
                            row["best_wait_baseline"],
                            f"{row['rl_wait_improvement_pct']:.2f}%",
                            row["best_queue_baseline"],
                            f"{row['rl_queue_improvement_pct']:.2f}%",
                        ]
                        for row in grid_rows
                    ],
                ),
            ]
        )
    markdown_parts.extend(
        [
            "",
            "## Reward Design Note",
            "Earlier reward-design experiments found queue-based and waiting-based rewards to give very similar results. The final report uses queue mode because it is simple and directly aligned with congestion reduction.",
            "",
            "CSV tables are saved under `tables/` and presentation figures under `figures/`.",
        ]
    )
    output_markdown.write_text("\n".join(markdown_parts) + "\n", encoding="utf-8")


def build_assets(results_root: Path) -> list[str]:
    figures_dir = results_root / "figures"
    tables_dir = results_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    final_summary = load_json(results_root / "final_double_dqn" / "dqn_multiseed_summary.json")
    original_summary = load_json(results_root / "original_double_dqn" / "dqn_multiseed_summary.json")
    ablation_summary = load_json(results_root / "ablations" / "ablation_summary.json")
    grid_summary_path = results_root / "grid_2x2" / "dqn_multiseed_summary.json"
    grid_summary = load_json(grid_summary_path) if grid_summary_path.exists() else None

    saved_figures: list[str] = []
    saved_figures.extend(
        plot_policy_dot(
            final_summary,
            metric="average_wait_time_seconds",
            title="Final RL Agent vs Baselines: Average Wait",
            xlabel="Average wait time (seconds)",
            output_base=figures_dir / "fig01_wait_rl_vs_baselines",
        )
    )
    saved_figures.extend(
        plot_policy_dot(
            final_summary,
            metric="average_queue_length",
            title="Final RL Agent vs Baselines: Average Queue",
            xlabel="Average queue length",
            output_base=figures_dir / "fig02_queue_rl_vs_baselines",
        )
    )
    saved_figures.extend(
        plot_best_baseline_improvement(
            final_summary,
            figures_dir / "fig03_best_baseline_vs_rl_improvement",
        )
    )

    regimes = list(aggregate_regimes(final_summary).keys())
    original_values = {
        metric: [dqn_metric(original_summary, regime, metric)[0] for regime in regimes]
        for metric in ["average_wait_time_seconds", "average_queue_length"]
    }
    tuned_values = {
        metric: [dqn_metric(final_summary, regime, metric)[0] for regime in regimes]
        for metric in ["average_wait_time_seconds", "average_queue_length"]
    }
    saved_figures.extend(
        plot_variant_comparison(
            regimes,
            original_values,
            tuned_values,
            "Original Double DQN",
            "Tuned Double DQN",
            "Original vs Tuned Double DQN",
            figures_dir / "fig04_original_vs_tuned_double_dqn",
        )
    )

    vanilla_values = {
        metric: [
            study_metric(ablation_summary, "algorithm_update", "vanilla_dqn", regime, metric)[0]
            for regime in regimes
        ]
        for metric in ["average_wait_time_seconds", "average_queue_length"]
    }
    double_values = {
        metric: [
            study_metric(ablation_summary, "algorithm_update", "double_dqn", regime, metric)[0]
            for regime in regimes
        ]
        for metric in ["average_wait_time_seconds", "average_queue_length"]
    }
    saved_figures.extend(
        plot_variant_comparison(
            regimes,
            vanilla_values,
            double_values,
            "Vanilla DQN",
            "Double DQN",
            "Double DQN Ablation",
            figures_dir / "fig05_double_dqn_ablation",
        )
    )

    saved_figures.extend(
        plot_action_mask_comparison(
            ablation_summary,
            figures_dir / "fig06_action_mask_ablation",
        )
    )
    saved_figures.extend(
        plot_training_curves(
            final_summary,
            figures_dir / "fig07_training_curves_final_dqn",
        )
    )
    if grid_summary is not None:
        saved_figures.extend(
            plot_policy_dot(
                grid_summary,
                metric="average_wait_time_seconds",
                title="2x2 Grid Double DQN vs Baselines: Average Wait",
                xlabel="Average wait time (seconds)",
                output_base=figures_dir / "fig08_2x2_wait_rl_vs_baselines",
            )
        )
        saved_figures.extend(
            plot_policy_dot(
                grid_summary,
                metric="average_queue_length",
                title="2x2 Grid Double DQN vs Baselines: Average Queue",
                xlabel="Average queue length",
                output_base=figures_dir / "fig09_2x2_queue_rl_vs_baselines",
            )
        )

    summarize_results(
        final_summary=final_summary,
        original_summary=original_summary,
        ablation_summary=ablation_summary,
        grid_summary=grid_summary,
        tables_dir=tables_dir,
        output_json=results_root / "final_statistics.json",
        output_markdown=results_root / "final_statistics.md",
    )
    return saved_figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=str,
        default="results/final_clean",
        help="Directory containing clean final experiment outputs.",
    )
    args = parser.parse_args()
    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = PROJECT_ROOT / results_root

    setup_style()
    saved_figures = build_assets(results_root)
    print(f"Saved final assets under {results_root}")
    for path in saved_figures:
        print(f"  {path}")


if __name__ == "__main__":
    main()
