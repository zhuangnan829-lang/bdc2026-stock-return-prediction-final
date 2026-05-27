# Experiment Leaderboard Report

## Scoring Rule

`stable_alpha_score = 0.25 * normalized(top5_return_mean) + 0.20 * normalized(cost_after_return) + 0.20 * normalized(sharpe) + 0.15 * normalized(rank_ic_mean) + 0.10 * normalized(worst_fold_rank_ic) - 0.05 * drawdown_penalty - 0.05 * turnover_penalty`

## Scan Scope

- `D:\Desktop\股票分析预测代码\app\model\experiments`
- `D:\Desktop\股票分析预测代码\app\model\weight_cap_search`
- `D:\Desktop\股票分析预测代码\app\model\weight_blend_search`
- `D:\Desktop\股票分析预测代码\app\model\turnover_stress_test`
- `D:\Desktop\股票分析预测代码\app\model\topk_objective_search`
- `D:\Desktop\股票分析预测代码\app\model\label_variant_search`
- `D:\Desktop\股票分析预测代码\app\model\rank_blend`
- `D:\Desktop\股票分析预测代码\app\model\sequence_length_search`
- `D:\Desktop\股票分析预测代码\app\model\transformer_lite`

CSV output: `D:\Desktop\股票分析预测代码\app\model\experiment_leaderboard.csv`

## Top 10 aggressive 候选

| candidate | score | decision | top5 | return | sharpe | worst_ic | max_dd | turnover | reason |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `clipped_return__topk_weighted_rank` | 0.608089 | adopt | 0.011344 | 1.714517 | 4.888188 | 0.021593 | -0.075602 | 0.953199 | 通过硬性规则 |
| `original_return__topk_weighted_rank` | 0.601354 | adopt | 0.012079 | 1.680136 | 4.790411 | 0.021528 | -0.086888 | 0.953216 | 通过硬性规则 |
| `residual_return__topk_weighted_rank` | 0.601348 | adopt | 0.012079 | 1.680136 | 4.790411 | 0.021525 | -0.086888 | 0.953216 | 通过硬性规则 |
| `topk30_gamma2_0` | 0.533480 | adopt | 0.012079 | 1.680136 | 4.790411 | -0.051537 | -0.086888 | 0.953216 | 通过硬性规则 |
| `alpha_1.0_cap_none` | 0.473914 | adopt | 0.029985 | 1.171246 | 4.019488 | n/a | -0.090067 | 0.956671 | 通过硬性规则 |
| `cap_0.25` | 0.473914 | adopt | 0.029985 | 1.171246 | 4.019488 | n/a | -0.090067 | 0.956671 | 通过硬性规则 |
| `cap_none` | 0.473914 | adopt | 0.029985 | 1.171246 | 4.019488 | n/a | -0.090067 | 0.956671 | 通过硬性规则 |
| `alpha_0.75_cap_none` | 0.473670 | adopt | 0.030014 | 1.169643 | 4.014978 | n/a | -0.090289 | 0.956558 | 通过硬性规则 |
| `alpha_0.5_cap_none` | 0.473452 | adopt | 0.030042 | 1.168098 | 4.010679 | n/a | -0.090487 | 0.956448 | 通过硬性规则 |
| `alpha_0.25_cap_none` | 0.473239 | adopt | 0.030070 | 1.166569 | 4.006415 | n/a | -0.090678 | 0.956339 | 通过硬性规则 |

## Top 10 robust 候选

| candidate | score | decision | top5 | return | sharpe | worst_ic | max_dd | turnover | reason |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `clipped_return__topk_weighted_rank` | 0.608089 | adopt | 0.011344 | 1.714517 | 4.888188 | 0.021593 | -0.075602 | 0.953199 | 通过硬性规则 |
| `original_return__topk_weighted_rank` | 0.601354 | adopt | 0.012079 | 1.680136 | 4.790411 | 0.021528 | -0.086888 | 0.953216 | 通过硬性规则 |
| `residual_return__topk_weighted_rank` | 0.601348 | adopt | 0.012079 | 1.680136 | 4.790411 | 0.021525 | -0.086888 | 0.953216 | 通过硬性规则 |
| `topk30_gamma2_0` | 0.533480 | adopt | 0.012079 | 1.680136 | 4.790411 | -0.051537 | -0.086888 | 0.953216 | 通过硬性规则 |
| `mt050_tc0010_pred_capnone` | 0.389556 | adopt | 0.009067 | 0.657082 | 4.061911 | n/a | -0.048777 | 0.500000 | 通过硬性规则 |
| `mt050_tc0010_blend_0.5_capnone` | 0.388941 | adopt | 0.009048 | 0.655281 | 4.057767 | n/a | -0.048883 | 0.500000 | 通过硬性规则 |
| `mt050_tc0010_blend_0.5_cap0.20` | 0.388308 | adopt | 0.009029 | 0.653410 | 4.053580 | n/a | -0.048990 | 0.500000 | 通过硬性规则 |
| `mt050_tc0010_equal_cap0.20` | 0.388308 | adopt | 0.009029 | 0.653410 | 4.053580 | n/a | -0.048990 | 0.500000 | 通过硬性规则 |
| `mt050_tc0010_equal_capnone` | 0.388308 | adopt | 0.009029 | 0.653410 | 4.053580 | n/a | -0.048990 | 0.500000 | 通过硬性规则 |
| `mt050_tc0010_pred_cap0.20` | 0.388308 | adopt | 0.009029 | 0.653410 | 4.053580 | n/a | -0.048990 | 0.500000 | 通过硬性规则 |

## 推荐最终候选

| candidate | score | decision | top5 | return | sharpe | worst_ic | max_dd | turnover | reason |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `clipped_return__topk_weighted_rank` | 0.608089 | adopt | 0.011344 | 1.714517 | 4.888188 | 0.021593 | -0.075602 | 0.953199 | 通过硬性规则 |

## 不推荐的高风险候选及原因

| candidate | score | decision | top5 | return | sharpe | worst_ic | max_dd | turnover | reason |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `topk20_gamma3_0` | 0.515324 | reject | 0.007469 | 1.690346 | 4.769483 | -0.053792 | -0.046822 | 0.940073 | top5_return_min_by_fold <= 0，缺少合理解释; worst_fold_rank_ic 明显低于当前 sl20 |
| `topk30_gamma3_0` | 0.488728 | reject | 0.009186 | 1.469555 | 4.457134 | -0.053146 | -0.082718 | 0.940236 | top5_return_min_by_fold <= 0，缺少合理解释 |
| `topk20_gamma5_0` | 0.484666 | reject | 0.004066 | 1.687721 | 4.588855 | -0.056716 | -0.079026 | 0.937923 | top5_return_min_by_fold <= 0，缺少合理解释; worst_fold_rank_ic 明显低于当前 sl20 |
| `topk30_gamma5_0` | 0.455201 | reject | 0.004468 | 1.552899 | 4.224830 | -0.054795 | -0.096007 | 0.951214 | top5_return_min_by_fold <= 0，缺少合理解释; worst_fold_rank_ic 明显低于当前 sl20 |
| `topk10_gamma3_0` | 0.442839 | reject | 0.003671 | 1.663163 | 4.804560 | -0.052146 | -0.031880 | 0.960467 | top5_return_min_by_fold <= 0，缺少合理解释 |
| `topk10_gamma5_0` | 0.424975 | reject | 0.000382 | 1.718410 | 4.614816 | -0.060337 | -0.070357 | 0.957888 | top5_return_min_by_fold <= 0，缺少合理解释; worst_fold_rank_ic 明显低于当前 sl20 |
| `topk20_gamma2_0` | 0.419357 | reject | 0.009203 | 1.372841 | 4.062365 | -0.050721 | -0.079311 | 0.955828 | top5_return_min_by_fold <= 0，缺少合理解释 |
| `v4_medium_lstm_sl40` | 0.418038 | reject | 0.010952 | 0.700974 | 2.605625 | 0.030504 | -0.156472 | 0.938396 | top5_return_min_by_fold <= 0，缺少合理解释; max_drawdown 明显恶化 |
| `topk10_gamma2_0` | 0.325392 | reject | 0.004628 | 1.114732 | 4.062836 | -0.044448 | -0.045112 | 0.951972 | top5_return_min_by_fold <= 0，缺少合理解释 |
| `sl60` | 0.316373 | reject | 0.010208 | 0.596388 | 2.481363 | -0.029453 | -0.099273 | 0.944684 | top5_return_min_by_fold <= 0，缺少合理解释 |
| `v4_medium_lstm_sl60` | 0.261468 | reject | 0.006669 | 0.505478 | 2.274359 | 0.013136 | -0.167002 | 0.906540 | top5_return_min_by_fold <= 0，缺少合理解释; max_drawdown 明显恶化 |
| `risk_adjusted_return__topk_weighted_rank` | 0.191805 | reject | 0.003789 | 0.566176 | 2.413197 | -0.006888 | -0.135120 | 0.943847 | max_drawdown 明显恶化 |
| `base_alpha_v4_medium` | 0.188307 | reject | 0.006252 | 0.320973 | n/a | 0.009668 | -0.130310 | 0.920797 | max_drawdown 明显恶化 |
| `topk5_gamma1_0` | 0.165032 | reject | 0.003485 | n/a | n/a | -0.024179 | n/a | n/a | top5_return_min_by_fold <= 0，缺少合理解释 |
| `sequence_length_search` | 0.077669 | reject | n/a | 0.700974 | 2.605625 | n/a | -0.156472 | 0.938396 | max_drawdown 明显恶化 |
| `label_variant_search` | 0.057837 | reject | n/a | 0.566176 | 2.413197 | n/a | -0.135120 | 0.943847 | max_drawdown 明显恶化 |
| `sl40` | 0.007374 | reject | 0.000112 | 0.244159 | 1.376108 | -0.059859 | -0.095101 | 0.925798 | top5_return_min_by_fold <= 0，缺少合理解释; worst_fold_rank_ic 明显低于当前 sl20 |
| `transformer_lite_sl60` | -0.077086 | reject | n/a | 0.111850 | 0.792688 | n/a | -0.121362 | 0.980408 | max_drawdown 明显恶化 |
| `transformer_lite_sl40` | -0.097275 | reject | n/a | 0.130101 | 0.793586 | n/a | -0.186779 | 0.994256 | max_drawdown 明显恶化 |

## Adoption Rules

- `top5_return_min_by_fold <= 0` 且没有合理解释，淘汰。
- `worst_fold_rank_ic` 明显低于当前 sl20，标记谨慎。
- `max_drawdown` 明显恶化，淘汰。
- `avg_turnover` 明显升高且收益没有补偿，淘汰。
- `single_slice_score` 高但 walk-forward 变差，标记疑似过拟合。
- `result_validator` 不通过，淘汰。
