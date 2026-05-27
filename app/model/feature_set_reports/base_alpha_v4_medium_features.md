# base_alpha_v4_medium Feature Report

## Purpose

`base_alpha_v4_medium` keeps the existing base/v3-style technical and crowding signals, then adds explicit reversal-protection and overheat-risk features. It is an experimental feature set only; it does not replace `base_alpha_v3_rs_crowding_mini4`.

## New Features

| feature | definition | window | leakage risk | missing/extreme handling |
|---|---|---:|---|---|
| `ret_2d` | current close vs close 2 trading days ago | 2d | No future data; uses current and past close only | inf/NaN replaced with 0 |
| `ret_7d` | current close vs close 7 trading days ago | 7d | No future data | inf/NaN replaced with 0 |
| `ret_15d` | current close vs close 15 trading days ago | 15d | No future data | inf/NaN replaced with 0 |
| `ret_20d` | current close vs close 20 trading days ago | 20d | No future data | inf/NaN replaced with 0 |
| `amount_ratio_10d` | current amount / rolling 10d mean amount - 1 | 10d | Rolling window includes current day and past only | denominator guarded by cleanup; inf/NaN to 0 |
| `close_position_10d` | close position inside rolling 10d high-low range | 10d | Current and past high/low only | denominator adds `1e-12`; inf/NaN to 0 |
| `close_position_20d` | close position inside rolling 20d high-low range | 20d | Current and past high/low only | denominator adds `1e-12`; inf/NaN to 0 |
| `ret_1d_zscore_cross_section` | daily cross-sectional z-score of `ret_1d` | same date | No future data; same-day cross-section only | clipped to [-5, 5], zero if std unavailable |
| `ret_3d_zscore_cross_section` | daily cross-sectional z-score of `ret_3d` | same date | No future data | clipped to [-5, 5], zero if std unavailable |
| `volume_spike_zscore` | daily cross-sectional z-score of `volume_ratio_5d` | 5d + same date | No future data | clipped to [-5, 5], zero if std unavailable |
| `turnover_spike_zscore` | daily cross-sectional z-score of `turnover_spike_5d` | 5d + same date | No future data | clipped to [-5, 5], zero if std unavailable |
| `overheat_score` | weighted positive short-return, volume-spike, and turnover-spike score | 1d/3d/5d | No future data | clipped to [0, 5], inf/NaN to 0 |
| `reversal_risk_score` | weighted `overheat_score`, volatility rank, and turnover rank | 5d/20d + same date | No future data | clipped to [0, 5], inf/NaN to 0 |
| `relative_to_market_5d` | stock `ret_5d` minus same-day market median `ret_5d` | 5d | No future data | inf/NaN to 0 |
| `relative_to_market_10d` | stock `ret_10d` minus same-day market median `ret_10d` | 10d | No future data | inf/NaN to 0 |

## Already Existing But Retained

`volatility_5d`, `volatility_10d`, `volatility_20d`, `volume_ratio_5d`, and `volume_ratio_10d` already existed in the feature pipeline and are retained in `base_alpha_v4_medium`.

## Leakage Notes

- Rolling features use pandas rolling windows on each stock ordered by date, with no negative shifts.
- Cross-sectional features use only stocks available on the same date.
- `future_open_1`, `future_open_5`, `target_return`, and training labels are excluded from the preset.
- Label construction still happens after feature engineering in train mode, so feature generation does not consume the future label columns.

## Missing Value Policy

After feature construction, all configured feature columns are sanitized by replacing `inf`, `-inf`, and missing values with 0. Cross-sectional z-scores use 0 when the same-day standard deviation is unavailable.
