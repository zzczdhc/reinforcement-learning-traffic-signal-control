"""Config loading and script smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_rl.config import load_config
from traffic_rl.experiments import train_and_evaluate_dqn


SMOKE_CONFIG = """
environment:
  train_schedule_name: stationary_smoke
  episode_length: 12
  step_seconds: 3
  min_green_time: 2
  yellow_time: 1
  max_departures_per_step: 3
  recent_arrival_window: 5
  observation_variant: full
  reward_mode: queue
  switch_penalty: 1.0
  train_schedule:
    - until_step: 12
      rates:
        N: 0.5
        S: 0.5
        E: 0.7
        W: 0.7
  evaluation_regimes:
    symmetric:
      - until_step: 12
        rates:
          N: 0.4
          S: 0.4
          E: 0.4
          W: 0.4
    asymmetric:
      - until_step: 12
        rates:
          N: 0.3
          S: 0.3
          E: 0.9
          W: 0.9

training:
  episodes: 2
  gamma: 0.95
  learning_rate: 0.001
  batch_size: 4
  buffer_size: 128
  hidden_dims: [16, 16]
  start_epsilon: 0.8
  end_epsilon: 0.1
  epsilon_decay_steps: 20
  warmup_steps: 2
  update_frequency: 1
  target_sync_steps: 4
  seed: 3

evaluation:
  episodes_per_regime: 2
"""


def build_smoke_config_dict() -> dict[str, object]:
    return {
        "environment": {
            "train_schedule_name": "stationary_smoke",
            "episode_length": 12,
            "step_seconds": 3,
            "min_green_time": 2,
            "yellow_time": 1,
            "max_departures_per_step": 3,
            "recent_arrival_window": 5,
            "observation_variant": "full",
            "reward_mode": "queue",
            "switch_penalty": 1.0,
            "train_schedule": [
                {"until_step": 12, "rates": {"N": 0.5, "S": 0.5, "E": 0.7, "W": 0.7}}
            ],
            "evaluation_regimes": {
                "symmetric": [
                    {"until_step": 12, "rates": {"N": 0.4, "S": 0.4, "E": 0.4, "W": 0.4}}
                ],
                "asymmetric": [
                    {"until_step": 12, "rates": {"N": 0.3, "S": 0.3, "E": 0.9, "W": 0.9}}
                ],
            },
        },
        "training": {
            "episodes": 2,
            "gamma": 0.95,
            "learning_rate": 0.001,
            "batch_size": 4,
            "buffer_size": 128,
            "hidden_dims": [16, 16],
            "start_epsilon": 0.8,
            "end_epsilon": 0.1,
            "epsilon_decay_steps": 20,
            "warmup_steps": 2,
            "update_frequency": 1,
            "target_sync_steps": 4,
            "seed": 3,
        },
        "evaluation": {
            "episodes_per_regime": 2,
        },
    }


class ConfigAndScriptSmokeTest(unittest.TestCase):
    def test_load_config_without_pyyaml_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(SMOKE_CONFIG, encoding="utf-8")

            config = load_config(config_path)

        self.assertEqual(config["environment"]["episode_length"], 12)
        self.assertEqual(config["training"]["hidden_dims"], [16, 16])
        self.assertEqual(config["evaluation"]["episodes_per_regime"], 2)

    def test_run_baselines_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            output_path = Path(tmpdir) / "baseline_summary.json"
            config_path.write_text(SMOKE_CONFIG, encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "run_baselines.py"),
                    "--config",
                    str(config_path),
                    "--output",
                    str(output_path),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIn("symmetric", payload)
        self.assertIn("max_pressure", payload["symmetric"])
        self.assertIn("Saved baseline summary", result.stdout)

    def test_run_training_script_and_summary_renderer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            checkpoint_path = Path(tmpdir) / "checkpoints" / "policy.pt"
            summary_path = Path(tmpdir) / "dqn_summary.json"
            config_path.write_text(SMOKE_CONFIG, encoding="utf-8")

            train_result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "train_dqn.py"),
                    "--config",
                    str(config_path),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--summary-output",
                    str(summary_path),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )

            render_result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "summarize_results.py"),
                    str(summary_path),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )

            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            checkpoint_exists = checkpoint_path.exists()

        self.assertTrue(checkpoint_exists)
        self.assertEqual(len(payload["training_history"]), 2)
        self.assertIn("dqn", payload["evaluation_results"]["symmetric"])
        self.assertIn("Saved checkpoint", train_result.stdout)
        self.assertIn("DQN summary", render_result.stdout)
        self.assertIn("train_schedule=stationary_smoke", render_result.stdout)

    def test_run_training_script_multiseed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            config_path = tmpdir_path / "config.yaml"
            multiseed_summary_path = tmpdir_path / "dqn_multiseed_summary.json"
            multiseed_output_dir = tmpdir_path / "multiseed"
            config_path.write_text(SMOKE_CONFIG, encoding="utf-8")

            train_result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "train_dqn.py"),
                    "--config",
                    str(config_path),
                    "--seeds",
                    "3,5",
                    "--multiseed-summary-output",
                    str(multiseed_summary_path),
                    "--multiseed-output-dir",
                    str(multiseed_output_dir),
                    "--no-plots",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )

            render_result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "summarize_results.py"),
                    str(multiseed_summary_path),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )

            payload = json.loads(multiseed_summary_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["seeds"], [3, 5])
        self.assertEqual(payload["metadata"]["run_count"], 2)
        self.assertTrue(payload["metadata"]["double_dqn"])
        self.assertEqual(len(payload["runs"]), 2)
        self.assertIn("aggregate", payload)
        self.assertIn("switch_frequency_per_step", payload["runs"][0]["final_training_episode"])
        self.assertIn("symmetric", payload["aggregate"]["per_regime"])
        self.assertIn("Saved multi-seed summary", train_result.stdout)
        self.assertIn("Multi-seed DQN summary", render_result.stdout)

    def test_train_and_evaluate_dqn_is_reproducible_for_same_seed(self) -> None:
        config = build_smoke_config_dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            payload_one = train_and_evaluate_dqn(
                config=config,
                checkpoint_path=tmpdir_path / "policy_one.pt",
                summary_path=tmpdir_path / "summary_one.json",
                verbose=False,
            )
            payload_two = train_and_evaluate_dqn(
                config=config,
                checkpoint_path=tmpdir_path / "policy_two.pt",
                summary_path=tmpdir_path / "summary_two.json",
                verbose=False,
            )

        self.assertEqual(payload_one["training_history"], payload_two["training_history"])
        self.assertEqual(payload_one["evaluation_results"], payload_two["evaluation_results"])

    def test_run_ablation_script_and_plotter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            base_config_path = tmpdir_path / "base_config.yaml"
            ablation_config_path = tmpdir_path / "ablations.yaml"
            output_root = tmpdir_path / "ablation_results"
            figure_dir = tmpdir_path / "figures"
            base_config_path.write_text(SMOKE_CONFIG, encoding="utf-8")
            ablation_config_path.write_text(
                f"""
base_config: {base_config_path}
global_defaults:
  output_root: {output_root}
  seeds: [5]
studies:
  algorithm_update:
    variants:
      vanilla_dqn:
        overrides:
          training:
            double_dqn: false
      double_dqn:
        overrides:
          training:
            double_dqn: true
  action_masking:
    variants:
      mask_off:
        overrides:
          training:
            use_action_mask: false
      mask_on:
        overrides:
          training:
            use_action_mask: true
""".strip(),
                encoding="utf-8",
            )

            ablation_result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "run_ablations.py"),
                    "--config",
                    str(ablation_config_path),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )

            summary_path = output_root / "ablation_summary.json"
            plot_result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "plot_ablations.py"),
                    str(summary_path),
                    "--output-dir",
                    str(figure_dir),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )

            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_exists = summary_path.exists()
            queue_plot_exists = (figure_dir / "baseline_vs_dqn_avg_queue.png").exists()
            double_dqn_plot_exists = (figure_dir / "double_dqn_ablation_avg_wait.png").exists()
            action_mask_plot_exists = (figure_dir / "action_mask_invalid_switch.png").exists()

        self.assertTrue(summary_exists)
        self.assertIn("algorithm_update", payload["studies"])
        self.assertIn("action_masking", payload["studies"])
        self.assertTrue(queue_plot_exists)
        self.assertTrue(double_dqn_plot_exists)
        self.assertTrue(action_mask_plot_exists)
        self.assertIn("Saved ablation summary", ablation_result.stdout)
        self.assertIn("Saved figures", plot_result.stdout)


if __name__ == "__main__":
    unittest.main()
