# Final Clean Run Statistics

## Overall Results
| Comparison | Wait improvement | Queue improvement | Other |
| --- | --- | --- | --- |
| Final RL vs best baseline | 7.55% | 7.39% | Averaged across regimes |
| Tuned vs original Double DQN | 1.10% | 1.02% | No new hyperparameter search |
| Double DQN vs vanilla DQN | 0.17% | 0.14% | Algorithm ablation |
| Action mask on vs off | 0.77% | 0.76% | Invalid switches reduced by 100.00% |
| 2x2 Double DQN vs best baseline | 6.04% | 5.83% | Extension experiment |

## Final RL Agent Overall Metrics
| Metric | Mean across regimes |
| --- | --- |
| Average wait (s) | 8.3601 |
| Average queue length | 9.4602 |
| Throughput per step | 3.2624 |
| Switch count | 47.61 |
| Invalid switch count | 0.00 |

## RL vs Best Baseline by Regime
| Regime | Best wait baseline | Wait improvement | Best queue baseline | Queue improvement |
| --- | --- | --- | --- | --- |
| symmetric_low | max_pressure | 4.36% | max_pressure | 4.19% |
| symmetric_high | queue_threshold | 7.93% | queue_threshold | 7.92% |
| asymmetric | max_pressure | 8.99% | max_pressure | 8.64% |
| nonstationary | max_pressure | 9.14% | max_pressure | 8.91% |
| burst_east_west | max_pressure | 7.35% | max_pressure | 7.31% |

## 2x2 Grid Extension
| Metric | Mean across grid regimes |
| --- | --- |
| Average wait (s) | 15.2588 |
| Average queue length | 24.7013 |
| Throughput per step | 4.7042 |
| Invalid switch count | 0.00 |

| Regime | Best wait baseline | Wait improvement | Best queue baseline | Queue improvement |
| --- | --- | --- | --- | --- |
| grid_balanced_low | max_pressure | 4.16% | max_pressure | 4.09% |
| grid_balanced_high | max_pressure | 6.29% | max_pressure | 5.99% |
| east_west_commute | max_pressure | 6.26% | max_pressure | 6.10% |
| north_south_commute | max_pressure | 4.57% | max_pressure | 4.34% |
| grid_nonstationary | max_pressure | 8.91% | max_pressure | 8.65% |

## Reward Design Note
Earlier reward-design experiments found queue-based and waiting-based rewards to give very similar results. The final report uses queue mode because it is simple and directly aligned with congestion reduction.

CSV tables are saved under `tables/` and presentation figures under `figures/`.
