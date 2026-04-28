# Results

This directory keeps the clean final artifacts used for the report and
presentation.

Final outputs live under `results/final_clean/`:

- `baselines/baseline_summary.json`: baseline policy evaluation
- `final_double_dqn/dqn_multiseed_summary.json`: final tuned Double DQN, seeds 7/17/27
- `original_double_dqn/dqn_multiseed_summary.json`: original Double DQN hyperparameters, seeds 7/17/27
- `ablations/ablation_summary.json`: core ablations for Double DQN and action masking
- `grid_2x2/baselines/baseline_summary.json`: 2x2 grid baseline evaluation
- `grid_2x2/dqn_multiseed_summary.json`: 2x2 grid Double DQN, seeds 7/17/27
- `figures/`: presentation-ready PNG/PDF figures
- `tables/`: CSV tables used by the figures and summary
- `final_statistics.md`: concise final numerical summary
- `final_statistics.json`: machine-readable final summary

The 2x2 grid extension figures are:

- `figures/fig08_2x2_wait_rl_vs_baselines.png`
- `figures/fig09_2x2_queue_rl_vs_baselines.png`

Regenerate the final assets after the experiments finish:

```bash
python3 scripts/build_final_assets.py --results-root results/final_clean
```

The report uses queue reward mode. Earlier reward-design experiments found
queue-based and waiting-time rewards to give very similar results, so the final
presentation keeps the simpler queue-based objective.
