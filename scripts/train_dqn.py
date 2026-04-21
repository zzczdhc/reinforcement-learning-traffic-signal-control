#!/usr/bin/env python3
"""Train a DQN controller and compare it against baseline policies."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_rl.config import load_config
from traffic_rl.experiments import train_and_evaluate_dqn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/checkpoints/dqn_policy.pt",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="results/dqn_summary.json",
        help="Path to save training and evaluation metrics.",
    )
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    output_path = PROJECT_ROOT / args.summary_output
    train_and_evaluate_dqn(
        config=config,
        checkpoint_path=checkpoint_path,
        summary_path=output_path,
        verbose=True,
    )


if __name__ == "__main__":
    main()
