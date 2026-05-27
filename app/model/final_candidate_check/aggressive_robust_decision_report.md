# Aggressive / Robust Candidate Decision

Generated: 2026-05-25

## Decision

- Aggressive: keep `app/model/default_submission_config.json` unchanged.
- Robust candidate: use `app/model/configs/submission_robust_regime_rerank_candidate.json`.
- Do not adopt snapshot ensemble as a replacement for the current single best checkpoint.

## Aggressive Profile

The aggressive profile remains the frozen default mainline:

- model: LSTM
- feature_set: `base_alpha_v3_rs_crowding_mini4`
- sequence_length: 20
- objective: `cross_section_rank`
- sort_strategy: `risk_adjusted`
- weight_strategy: `pred`
- top_k: 5
- candidate_size: 180
- transaction_cost: 0.001
- max_single_weight: 0.18

Latest default replay wrote `app/output/result.csv` and passed `result_validator.py`.

Selected stocks:

| stock_id | weight |
| --- | ---: |
| 300316 | 0.18 |
| 600183 | 0.18 |
| 600584 | 0.18 |
| 601877 | 0.18 |
| 688396 | 0.18 |

## Robust Candidate

The robust candidate keeps the same model, feature set, ranking strategy, and weighting strategy as the default profile, then adds a simple regime-aware rerank rule:

- regime flag: `is_high_volatility`
- signal: `close_position_20d`
- rerank weight: -0.05
- low-volatility dates: keep default aggressive ranking

Latest robust replay used:

`app/model/configs/submission_robust_regime_rerank_candidate.json`

and wrote:

`app/model/final_candidate_check/robust_regime_result.csv`

The current prediction date `2026-03-13` was detected as `high_volatility_range`, so the robust rerank rule was active. The replay passed `result_validator.py`.

Selected stocks:

| stock_id | weight |
| --- | ---: |
| 300316 | 0.18 |
| 600115 | 0.18 |
| 600183 | 0.18 |
| 600584 | 0.18 |
| 688396 | 0.18 |

## Evidence

Regime-aware fusion comparison favored `hv_close_position_20d_m005` as the robust candidate:

| profile | cost_after_return | sharpe | max_drawdown | selected_top5_return_mean | fold3_selected_top5_return | high_volatility_selected_top5_return | poor_false_positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1.085936 | 3.977455 | -0.087064 | 0.016264 | 0.018838 | 0.018838 | 128 |
| hv_close_position_20d_m005 | 1.191992 | 4.165041 | -0.084420 | 0.017184 | 0.021600 | 0.021600 | 125 |

The robust candidate improves the stressed high-volatility slice while leaving low-volatility dates on the aggressive path.

## Snapshot Ensemble Check

Snapshot average-rank ensemble was tested under the same LSTM setup. It did not replace the best single checkpoint:

| profile | rank_ic_mean | worst_fold_rank_ic | top5_return_mean | cost_after_return | Sharpe | max_drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| snapshot_rank1 | 0.027982 | -0.033492 | 0.007502 | 1.085936 | 3.977455 | -0.087064 |
| snapshot_top3_average_rank | -0.011819 | -0.057413 | 0.002546 | 0.442818 | 2.466286 | -0.063780 |

Conclusion: the last epoch can be worse than the best checkpoint, but top-3 snapshot rank averaging weakens RankIC and return too much. Keep best-checkpoint inference.

## Freeze / Validation Status

Completed checks:

- `python app/code/src/compare_config_consistency.py`: passed, 58 checks, 0 failed.
- `python app/code/src/cli.py predict`: passed and wrote default `app/output/result.csv`.
- robust replay with `SUBMISSION_CONFIG_PATH=app/model/configs/submission_robust_regime_rerank_candidate.json`: passed and wrote `app/model/final_candidate_check/robust_regime_result.csv`.
- `python app/code/src/cli.py freeze`: passed config consistency, inference, result validation, and pre-submit checks.

## Next Step

Use the default aggressive profile for formal submission unless the user explicitly chooses the robust candidate for a risk-controlled run. If preparing a final package, keep both files:

- `app/model/default_submission_config.json`
- `app/model/configs/submission_robust_regime_rerank_candidate.json`

Do not overwrite the default config with the robust candidate without an explicit final decision.
