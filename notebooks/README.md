# Notebook Status

No project notebooks are committed yet.

Recommended first notebooks after the runnable core is stable:

1. `01_simulator_sanity_check.ipynb`
   Validate queue growth, departures, and switching behavior under simple schedules.
2. `02_baseline_comparison.ipynb`
   Load `results/baseline_summary.json` and compare the heuristic controllers across regimes.
3. `03_dqn_training_analysis.ipynb`
   Load `results/dqn_summary.json`, inspect training history, and compare DQN against baselines.
4. `04_ablation_analysis.ipynb`
   Load `results/ablations/ablation_summary.json` and annotate the generated figures for slides.

Use the scripts first:

- `scripts/run_ablations.py` for batch experiments
- `scripts/plot_ablations.py` for presentation-ready figures

Treat notebooks as optional analysis and annotation layers on top of the JSON outputs.
