# Config Consistency Report

- Status: PASS
- Checks: 58
- Failed: 0
- Authoritative source: `app/model/default_submission_config.json`

## Required Unified Fields

| field | source |
|---|---|
| model_name | default_submission_config.model_family |
| feature_set | default_submission_config.feature_set |
| sequence_length | validation_scheme.sequence_length |
| sort_strategy | selection_logic.sort_strategy |
| weight_strategy | selection_logic.weighting_scheme |
| top_k | selection_logic.top_k |
| candidate_size | selection_logic.primary_candidate_size |
| risk_penalty_weight | risk_filter_thresholds.risk_penalty_weight |
| max_turnover | execution_logic.max_turnover |
| transaction_cost | execution_logic.transaction_cost |
| max_single_weight | selection_logic.max_single_weight, nullable |

## Failed Checks

No inconsistencies found.

## All Checks

| check | status | expected | actual |
|---|---:|---|---|
| profile_name: default vs best | PASS | `lstm_sl20_base_alpha_v3_rs_crowding_mini4__hv_close_position_rerank` | `lstm_sl20_base_alpha_v3_rs_crowding_mini4__hv_close_position_rerank` |
| profile_name: default vs model_meta | PASS | `lstm_sl20_base_alpha_v3_rs_crowding_mini4__hv_close_position_rerank` | `lstm_sl20_base_alpha_v3_rs_crowding_mini4__hv_close_position_rerank` |
| training.feature_set: default vs best | PASS | `base_alpha_v3_rs_crowding_mini4` | `base_alpha_v3_rs_crowding_mini4` |
| training.target_mode: default vs best | PASS | `cross_section_rank` | `cross_section_rank` |
| training.model_family: default vs best | PASS | `lstm` | `lstm` |
| training.seed: default vs best | PASS | `2026` | `2026` |
| training.valid_dates: default vs best | PASS | `20` | `20` |
| training.num_folds: default vs best | PASS | `3` | `3` |
| training.sequence_length: default vs best | PASS | `20` | `20` |
| feature_set: default vs model_meta | PASS | `base_alpha_v3_rs_crowding_mini4` | `base_alpha_v3_rs_crowding_mini4` |
| target_mode: default vs model_meta | PASS | `cross_section_rank` | `cross_section_rank` |
| model_family: default vs model_meta | PASS | `lstm` | `lstm` |
| seed: default vs best | PASS | `2026` | `2026` |
| seed: default vs model_meta | PASS | `2026` | `2026` |
| sequence_length: default vs model_meta | PASS | `20` | `20` |
| selection.top_k: default vs best | PASS | `5` | `5` |
| selection.top_k: default vs model_meta | PASS | `5` | `5` |
| selection.primary_candidate_size: default vs best | PASS | `180` | `180` |
| selection.primary_candidate_size: default vs model_meta | PASS | `180` | `180` |
| selection.enable_risk_filters: default vs best | PASS | `True` | `True` |
| selection.enable_risk_filters: default vs model_meta | PASS | `True` | `True` |
| selection.sort_strategy: default vs best | PASS | `risk_adjusted` | `risk_adjusted` |
| selection.sort_strategy: default vs model_meta | PASS | `risk_adjusted` | `risk_adjusted` |
| selection.weighting_scheme: default vs best | PASS | `pred` | `pred` |
| selection.weighting_scheme: default vs model_meta | PASS | `pred` | `pred` |
| selection.max_single_weight: default vs best | PASS | `0.18` | `0.18` |
| selection.max_single_weight: default vs model_meta | PASS | `0.18` | `0.18` |
| risk.max_volatility_20d_pct: default vs best | PASS | `0.86` | `0.86` |
| risk.max_volatility_20d_pct: default vs model_meta | PASS | `0.86` | `0.86` |
| risk.max_volatility_5d_pct: default vs best | PASS | `1.0` | `1.0` |
| risk.max_volatility_5d_pct: default vs model_meta | PASS | `1.0` | `1.0` |
| risk.turnover_rate_lower_pct: default vs best | PASS | `0.03` | `0.03` |
| risk.turnover_rate_lower_pct: default vs model_meta | PASS | `0.03` | `0.03` |
| risk.turnover_rate_upper_pct: default vs best | PASS | `0.97` | `0.97` |
| risk.turnover_rate_upper_pct: default vs model_meta | PASS | `0.97` | `0.97` |
| risk.turnover_ratio_upper_pct: default vs best | PASS | `0.95` | `0.95` |
| risk.turnover_ratio_upper_pct: default vs model_meta | PASS | `0.95` | `0.95` |
| risk.risk_penalty_weight: default vs best | PASS | `-0.3` | `-0.3` |
| risk.risk_penalty_weight: default vs model_meta | PASS | `-0.3` | `-0.3` |
| execution.use_previous_result_when_available: default vs best | PASS | `False` | `False` |
| execution.use_previous_result_when_available: default vs model_meta | PASS | `False` | `False` |
| execution.max_turnover: default vs best | PASS | `1.0` | `1.0` |
| execution.max_turnover: default vs model_meta | PASS | `1.0` | `1.0` |
| execution.transaction_cost: default vs best | PASS | `0.001` | `0.001` |
| execution.transaction_cost: default vs model_meta | PASS | `0.001` | `0.001` |
| loader.top_k vs unified top_k | PASS | `5` | `5` |
| loader.candidate_size vs unified candidate_size | PASS | `180` | `180` |
| loader.weight_strategy vs unified weight_strategy | PASS | `pred` | `pred` |
| loader.risk_penalty_weight vs unified risk_penalty_weight | PASS | `-0.3` | `-0.3` |
| loader.max_turnover vs unified max_turnover | PASS | `1.0` | `1.0` |
| loader.transaction_cost vs unified transaction_cost | PASS | `0.001` | `0.001` |
| script.run_submission.sh uses config consistency guard | PASS | `ok` | `missing compare_config_consistency.py` |
| script.test.sh uses config consistency guard | PASS | `ok` | `missing compare_config_consistency.py` |
| script.test.ps1 uses config consistency guard | PASS | `ok` | `missing compare_config_consistency.py` |
| script.freeze_submission.sh uses config consistency guard | PASS | `ok` | `missing compare_config_consistency.py` |
| script.cli.py uses config consistency guard | PASS | `ok` | `missing build_default_inference_args` |
| script.test.sh has no legacy parameter defaults | PASS | `ok` | `ok` |
| script.test.ps1 has no legacy parameter defaults | PASS | `ok` | `ok` |
