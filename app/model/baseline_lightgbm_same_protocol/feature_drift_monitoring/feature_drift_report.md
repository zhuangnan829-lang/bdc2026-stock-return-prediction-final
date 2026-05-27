# Feature Drift and Importance Report

## Scope

- Validation folds: 1, 2, 3
- Features analyzed: 20
- Drift score combines fold-to-fold mean shift, median shift, missing-rate change, and stage IC range.

## Fold Overview

| Fold | Feature Count | Avg Abs IC | Avg Missing Rate |
|---:|---:|---:|---:|
| 1 | 20 | 0.024038 | 0.000000 |
| 2 | 20 | 0.047442 | 0.000000 |
| 3 | 20 | 0.031682 | 0.000000 |

## Stable Features

These features have relatively small distribution movement across Fold 1/2/3 and comparatively stable stage IC.

| Feature | drift_score | stage_ic_mean | stage_ic_range | relative_mean_range | importance_rank |
|---|---:|---:|---:|---:|---:|
| `volume_change_1d` | 0.167063 | -0.011123 | 0.010121 | 0.037543 | 15 |
| `ret_1d` | 0.211616 | -0.008367 | 0.010624 | 0.076707 | 11 |
| `amount_change_1d` | 0.181400 | -0.010768 | 0.010870 | 0.040498 | 13 |
| `intraday_return` | 0.483458 | -0.017609 | 0.037566 | 0.072624 | 14 |
| `turnover_spike_5d` | 0.727354 | -0.010537 | 0.042737 | 0.200552 | 17 |
| `volume_ratio_5d` | 0.729737 | -0.011089 | 0.042747 | 0.202492 | 18 |
| `ret_3d` | 0.643103 | 0.006855 | 0.047019 | 0.140253 | 6 |

## Unstable Features

These features show larger fold-to-fold drift or IC reversal, so they should be watched when Fold 1/3 weakens.

| Feature | drift_score | stage_ic_mean | stage_ic_range | relative_mean_range | importance_rank |
|---|---:|---:|---:|---:|---:|
| `mom_10d` | 2.504328 | 0.005676 | 0.190625 | 0.444333 | 19 |
| `ret_10d` | 2.504328 | 0.005676 | 0.190625 | 0.444333 | 3 |
| `range_pct` | 1.979660 | 0.027518 | 0.155982 | 0.310328 | 4 |
| `volume_price_divergence_5d` | 1.374524 | 0.001458 | 0.136697 | 0.000000 | 7 |
| `close_to_low` | 1.315303 | 0.007546 | 0.114084 | 0.152549 | 9 |
| `rel_cs_mean_close_to_ma_10d` | 1.292658 | 0.002817 | 0.122694 | 0.000000 | 5 |

## Important But Unstable

High-importance features in this list can explain why a fold fails even when the average model score looks acceptable.

| Feature | importance_rank | gain_importance_pct | drift_score | stage_ic_range |
|---|---:|---:|---:|---:|
| `ret_10d` | 3 | 0.077207 | 2.504328 | 0.190625 |
| `range_pct` | 4 | 0.074214 | 1.979660 | 0.155982 |
| `rel_cs_mean_close_to_ma_10d` | 5 | 0.072881 | 1.292658 | 0.122694 |
| `volume_price_divergence_5d` | 7 | 0.058546 | 1.374524 | 0.136697 |
| `close_to_low` | 9 | 0.054367 | 1.315303 | 0.114084 |
| `mom_10d` | 19 | 0.013825 | 2.504328 | 0.190625 |

## Top LightGBM Importance

| Feature | gain_importance_pct | split_importance_pct | stage_ic_mean | stage_ic_range |
|---|---:|---:|---:|---:|
| `ret_5d` | 0.087581 | 0.071917 | -0.008896 | 0.092556 |
| `rel_hs300_mean_ret_5d` | 0.079780 | 0.061000 | -0.008896 | 0.092556 |
| `ret_10d` | 0.077207 | 0.074333 | 0.005676 | 0.190625 |
| `range_pct` | 0.074214 | 0.070000 | 0.027518 | 0.155982 |
| `rel_cs_mean_close_to_ma_10d` | 0.072881 | 0.062833 | 0.002817 | 0.122694 |
| `ret_3d` | 0.064501 | 0.063500 | 0.006855 | 0.047019 |
| `volume_price_divergence_5d` | 0.058546 | 0.064417 | 0.001458 | 0.136697 |
| `volume_change_5d` | 0.055035 | 0.061917 | -0.012488 | 0.080677 |
| `close_to_low` | 0.054367 | 0.058917 | 0.007546 | 0.114084 |
| `close_to_high` | 0.054098 | 0.057167 | -0.031984 | 0.080947 |
| `ret_1d` | 0.050880 | 0.054000 | -0.008367 | 0.010624 |
| `volume_ratio_10d` | 0.041689 | 0.046167 | -0.007815 | 0.053307 |

## Interpretation

- Stable features are better candidates for the core signal set because their fold IC and cross-sectional distribution are less regime-dependent.
- Unstable features are not automatically bad, but they need caps, ablation checks, or regime-aware usage if they also rank high in LightGBM importance.
- If Fold 1 or Fold 3 underperforms, first inspect important-but-unstable features with large `stage_ic_range`; those are the most likely contributors to signal failure.