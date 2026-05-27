# Full Optimization Validation Report

Report path: `D:\Desktop\股票分析预测代码\app\model\full_optimization_validation_report.md`
Log directory: `D:\Desktop\股票分析预测代码\app\model\full_optimization_validation_logs`

## Pipeline Status

| step | status | seconds | error |
|---|---|---:|---|
| compare_config_consistency | PASS | 0.2 |  |
| analyze_performance_bottleneck | PASS | 4.0 |  |
| search_weight_cap | PASS | 6.0 |  |
| search_weight_blend | PASS | 10.9 |  |
| turnover_stress_test | PASS | 226.1 |  |
| evaluate_rank_stability | PASS | 1.5 |  |
| diagnose_misranked_samples | PASS | 3.9 |  |
| evaluate_by_market_regime | PASS | 15.0 |  |
| regime_rerank_switch | PASS | 31.9 |  |
| rank_blend | PASS | 23.9 |  |
| build_experiment_leaderboard | PASS | 1.5 |  |
| select_final_submission_config | PASS | 0.6 |  |
| result_validator | PASS | 0.6 |  |
| pre_submit_check | PASS | 0.6 |  |

## Failed Steps

- 无失败步骤。

## 1. 当前主线表现

- profile: `lstm_sl20_base_alpha_v3_rs_crowding_mini4__hv_close_position_rerank`
- feature_set: `base_alpha_v3_rs_crowding_mini4`
- model_family: `lstm`
- result validation: rows=5, weight_sum=0.900000, stocks=300316, 600115, 600183, 600584, 688396
- walk-forward rank_ic_mean: `0.027982`
- walk-forward top5_mean_return_mean: `0.007502`
- backtest return: `1.171246`
- backtest sharpe: `4.019488`
- backtest max_drawdown: `-0.090067`
- backtest avg_turnover: `0.956671`

## 2. 收益瓶颈判断

| date | true_top5_return | candidate_pool_true_top5_return | model_top5_equal_return | model_top5_weighted_return | model_top5_hit_rate | model_top10_hit_rate | worst_stock_return_in_model_top5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-11-26 | 0.127572 | 0.087239 | 0.009297 | 0.008367 | 0.000000 | 0.000000 | -0.001715 |
| 2025-11-27 | 0.139344 | 0.125000 | 0.002906 | 0.002615 | 0.000000 | 0.000000 | -0.042963 |
| 2025-11-28 | 0.120847 | 0.063285 | -0.008998 | -0.008098 | 0.000000 | 0.000000 | -0.041569 |
| 2025-12-01 | 0.136380 | 0.127373 | 0.022308 | 0.020078 | 0.000000 | 0.000000 | -0.018178 |
| 2025-12-02 | 0.170168 | 0.079847 | 0.036206 | 0.032585 | 0.000000 | 0.000000 | 0.007499 |

## 3. 组合层优化结果

Weight cap top rows:
| cap | single_slice_score | cost_after_return | sharpe | max_drawdown | avg_turnover |
| --- | --- | --- | --- | --- | --- |
| none | 0.029985 | 1.171246 | 4.019488 | -0.090067 | 0.956671 |
| 0.25 | 0.029985 | 1.171246 | 4.019488 | -0.090067 | 0.956671 |
| 0.20 | 0.030099 | 1.165010 | 4.002179 | -0.090852 | 0.956291 |
| 0.18 | 0.027089 | 1.085936 | 3.977455 | -0.087064 | 0.931137 |
| 0.16 | 0.024079 | 0.999800 | 3.930683 | -0.082412 | 0.898850 |
| 0.14 | 0.021069 | 0.859260 | 3.891290 | -0.077989 | 0.834677 |

Weight blend top rows:
| alpha | max_single_weight | single_slice_score | cost_after_return | sharpe | max_drawdown | avg_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| 1.000000 | none | 0.029985 | 1.171246 | 4.019488 | -0.090067 | 0.956671 |
| 0.750000 | none | 0.030014 | 1.169643 | 4.014978 | -0.090289 | 0.956558 |
| 0.500000 | none | 0.030042 | 1.168098 | 4.010679 | -0.090487 | 0.956448 |
| 0.250000 | none | 0.030070 | 1.166569 | 4.006415 | -0.090678 | 0.956339 |
| 0.000000 | 0.20 | 0.030099 | 1.165010 | 4.002179 | -0.090852 | 0.956291 |
| 0.000000 | none | 0.030099 | 1.165010 | 4.002179 | -0.090852 | 0.956291 |

Turnover stress robust rows:
| profile_name | cost_after_return | sharpe | max_drawdown | avg_turnover | robust_score |
| --- | --- | --- | --- | --- | --- |
| mt050_tc0010_pred_capnone | 0.657082 | 4.061911 | -0.048777 | 0.500000 | 1.286406 |
| mt050_tc0010_blend_0.5_capnone | 0.655281 | 4.057767 | -0.048883 | 0.500000 | 1.278876 |
| mt050_tc0010_blend_0.5_cap0.20 | 0.653410 | 4.053580 | -0.048990 | 0.500000 | 1.271209 |
| mt050_tc0010_equal_cap0.20 | 0.653410 | 4.053580 | -0.048990 | 0.500000 | 1.271209 |
| mt050_tc0010_pred_cap0.20 | 0.653410 | 4.053580 | -0.048990 | 0.500000 | 1.271209 |
| mt050_tc0010_equal_capnone | 0.653410 | 4.053580 | -0.048990 | 0.500000 | 1.271209 |

## 4. 稳定性优化结果

| model | feature_set | sequence_length | rank_ic_mean | worst_fold_rank_ic | top5_return_mean | stability_score |
| --- | --- | --- | --- | --- | --- | --- |
| lstm | base_alpha_v3_rs_crowding_mini4 | 20.000000 | 0.027982 | -0.033492 | 0.007502 | -0.037332 |

## 5. 标签/特征/融合实验结果

Rank blend:
| blend_name | requested_components | used_components | missing_components | rank_ic_mean | worst_fold_rank_ic | top5_return_mean | top5_return_min_by_fold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_lstm_single | {"lstm": 1.0} | lstm | n/a | 0.027982 | -0.033492 | 0.007502 | 0.003244 |
| A_lstm70_lightgbm30 | {"lightgbm": 0.3, "lstm": 0.7} | lstm,lightgbm | n/a | 0.017071 | -0.049245 | 0.004544 | -0.000265 |
| B_lstm60_lightgbm20_momentum20 | {"lightgbm": 0.2, "lstm": 0.6, "momentum": 0.2} | lstm,lightgbm,momentum | n/a | 0.016463 | -0.052472 | 0.001606 | -0.005790 |
| C_lstm50_lightgbm30_reversal20 | {"lightgbm": 0.3, "lstm": 0.5, "reversal": 0.2} | lstm,lightgbm,reversal | n/a | 0.005549 | -0.057183 | 0.002515 | -0.002123 |
| D_lstm50_lightgbm25_xgboost25 | {"lightgbm": 0.25, "lstm": 0.5, "xgboost": 0.25} | lstm,lightgbm,xgboost | n/a | 0.001854 | -0.059020 | 0.002164 | -0.002833 |
| E_lstm60_momentum40 | {"lstm": 0.6, "momentum": 0.4} | lstm,momentum | n/a | 0.018159 | -0.044200 | 0.002334 | -0.007603 |

Regime rerank:
| profile_name | cost_after_return | sharpe | max_drawdown | avg_turnover | delta_cost_after_return | delta_sharpe | delta_high_volatility_selected_top5_return |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1.085936 | 3.977455 | -0.087064 | 0.931137 | 0.000000 | 0.000000 | 0.000000 |
| global_close_position_20d_m005 | 1.153479 | 4.033127 | -0.084423 | 0.931303 | 0.067542 | 0.055672 | 0.002761 |
| global_reversal_risk_score_m005 | 1.085916 | 4.059159 | -0.074553 | 0.931864 | -0.000020 | 0.081704 | 0.000576 |
| hv_close_position_20d_m005 | 1.191992 | 4.165041 | -0.084420 | 0.929450 | 0.106056 | 0.187586 | 0.002761 |
| hv_reversal_risk_score_m005 | 1.085701 | 4.043042 | -0.074324 | 0.917193 | -0.000235 | 0.065587 | 0.000576 |
| hvrange_close_position_20d_m005 | 1.195231 | 4.155355 | -0.087064 | 0.929427 | 0.109295 | 0.177900 | 0.002480 |
| hvrange_reversal_risk_score_m005 | 1.106581 | 4.127425 | -0.068898 | 0.917209 | 0.020645 | 0.149971 | 0.000377 |
| hv_combo_m005 | 1.054269 | 4.117165 | -0.084420 | 0.931314 | -0.031667 | 0.139710 | -0.000293 |

Market regime:
| stage_id | stage_name | sample_days | prediction_rows | rank_ic | top5_mean_return | backtest_return | avg_turnover |
| --- | --- | --- | --- | --- | --- | --- | --- |
| low_volatility | 低波动 | 40.000000 | 12000.000000 | 0.042727 | 0.009632 | 0.568672 | 0.967328 |
| high_volatility | 高波动 | 20.000000 | 6000.000000 | -0.001508 | 0.003244 | 0.384130 | 0.935358 |
| trend | 趋势 | 31.000000 | 9300.000000 | 0.039054 | 0.014351 | 0.444240 | 0.952376 |
| range | 震荡 | 29.000000 | 8700.000000 | 0.016146 | 0.000182 | 0.503383 | 0.961263 |
| high_volatility_range | 高波动震荡 | 8.000000 | 2400.000000 | -0.007992 | -0.006384 | 0.258428 | 0.932402 |

Leaderboard top rows:
| rank | candidate_label | stable_alpha_score | decision | top5_mean_return | cumulative_return_after_cost | sharpe_after_cost | risk_flags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000000 | clipped_return__topk_weighted_rank | 0.608089 | adopt | 0.011344 | 1.714517 | 4.888188 | 通过硬性规则 |
| 2.000000 | original_return__topk_weighted_rank | 0.601354 | adopt | 0.012079 | 1.680136 | 4.790411 | 通过硬性规则 |
| 3.000000 | residual_return__topk_weighted_rank | 0.601348 | adopt | 0.012079 | 1.680136 | 4.790411 | 通过硬性规则 |
| 4.000000 | topk30_gamma2_0 | 0.533480 | adopt | 0.012079 | 1.680136 | 4.790411 | 通过硬性规则 |
| 5.000000 | topk20_gamma3_0 | 0.515324 | reject | 0.007469 | 1.690346 | 4.769483 | top5_return_min_by_fold <= 0，缺少合理解释; worst_fold_rank_ic 明显低于当前 sl20 |
| 6.000000 | topk30_gamma3_0 | 0.488728 | reject | 0.009186 | 1.469555 | 4.457134 | top5_return_min_by_fold <= 0，缺少合理解释 |
| 7.000000 | topk20_gamma5_0 | 0.484666 | reject | 0.004066 | 1.687721 | 4.588855 | top5_return_min_by_fold <= 0，缺少合理解释; worst_fold_rank_ic 明显低于当前 sl20 |
| 8.000000 | alpha_1.0_cap_none | 0.473914 | adopt | 0.029985 | 1.171246 | 4.019488 | 通过硬性规则 |

## 6. aggressive 候选

| aggressive_rank | candidate_label | stable_alpha_score | decision | top5_mean_return | cumulative_return_after_cost | sharpe_after_cost | risk_flags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000000 | clipped_return__topk_weighted_rank | 0.608089 | adopt | 0.011344 | 1.714517 | 4.888188 | 通过硬性规则 |
| 2.000000 | original_return__topk_weighted_rank | 0.601354 | adopt | 0.012079 | 1.680136 | 4.790411 | 通过硬性规则 |
| 3.000000 | residual_return__topk_weighted_rank | 0.601348 | adopt | 0.012079 | 1.680136 | 4.790411 | 通过硬性规则 |
| 4.000000 | topk30_gamma2_0 | 0.533480 | adopt | 0.012079 | 1.680136 | 4.790411 | 通过硬性规则 |
| 5.000000 | alpha_1.0_cap_none | 0.473914 | adopt | 0.029985 | 1.171246 | 4.019488 | 通过硬性规则 |
| 6.000000 | cap_0.25 | 0.473914 | adopt | 0.029985 | 1.171246 | 4.019488 | 通过硬性规则 |
| 7.000000 | cap_none | 0.473914 | adopt | 0.029985 | 1.171246 | 4.019488 | 通过硬性规则 |
| 8.000000 | alpha_0.75_cap_none | 0.473670 | adopt | 0.030014 | 1.169643 | 4.014978 | 通过硬性规则 |
| 9.000000 | alpha_0.5_cap_none | 0.473452 | adopt | 0.030042 | 1.168098 | 4.010679 | 通过硬性规则 |
| 10.000000 | alpha_0.25_cap_none | 0.473239 | adopt | 0.030070 | 1.166569 | 4.006415 | 通过硬性规则 |

## 7. robust 候选

| robust_rank | candidate_label | stable_alpha_score | decision | max_drawdown_after_cost | avg_turnover | cumulative_return_after_cost | risk_flags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000000 | clipped_return__topk_weighted_rank | 0.608089 | adopt | -0.075602 | 0.953199 | 1.714517 | 通过硬性规则 |
| 2.000000 | original_return__topk_weighted_rank | 0.601354 | adopt | -0.086888 | 0.953216 | 1.680136 | 通过硬性规则 |
| 3.000000 | residual_return__topk_weighted_rank | 0.601348 | adopt | -0.086888 | 0.953216 | 1.680136 | 通过硬性规则 |
| 4.000000 | topk30_gamma2_0 | 0.533480 | adopt | -0.086888 | 0.953216 | 1.680136 | 通过硬性规则 |
| 5.000000 | mt050_tc0010_pred_capnone | 0.389556 | adopt | -0.048777 | 0.500000 | 0.657082 | 通过硬性规则 |
| 6.000000 | mt050_tc0010_blend_0.5_capnone | 0.388941 | adopt | -0.048883 | 0.500000 | 0.655281 | 通过硬性规则 |
| 7.000000 | mt050_tc0010_blend_0.5_cap0.20 | 0.388308 | adopt | -0.048990 | 0.500000 | 0.653410 | 通过硬性规则 |
| 8.000000 | mt050_tc0010_equal_cap0.20 | 0.388308 | adopt | -0.048990 | 0.500000 | 0.653410 | 通过硬性规则 |
| 9.000000 | mt050_tc0010_equal_capnone | 0.388308 | adopt | -0.048990 | 0.500000 | 0.653410 | 通过硬性规则 |
| 10.000000 | mt050_tc0010_pred_cap0.20 | 0.388308 | adopt | -0.048990 | 0.500000 | 0.653410 | 通过硬性规则 |

## 8. 是否建议替换 LSTM sl20

否。未证明任何方案在 walk-forward、回测、单切片三方面同时优于 sl20。

## 9. 最终推荐

保留 LSTM sl20 主线；aggressive/robust 配置只作为最终提交目标不同的候选，不直接替换主线。 `mt050_tc0010_blend_0.5_cap0.20` 可进入候选观察：max_drawdown=-0.048990, avg_turnover=0.500000, return=0.653410。 `hv_close_position_20d_m005` 可作为高波动轻重排候选：delta_return=0.106056, delta_sharpe=0.187586, delta_max_drawdown=0.002645, delta_high_vol=0.002761。

## 10. 仍然存在的风险

- 主要风险仍是单票贡献集中、换手偏高、部分 TopK/新模型 worst fold 或 fold 内 Top5 最差收益不稳定。
- 高风险候选如下：
| candidate_label | stable_alpha_score | risk_flags |
| --- | --- | --- |
| topk20_gamma3_0 | 0.515324 | top5_return_min_by_fold <= 0，缺少合理解释; worst_fold_rank_ic 明显低于当前 sl20 |
| topk30_gamma3_0 | 0.488728 | top5_return_min_by_fold <= 0，缺少合理解释 |
| topk20_gamma5_0 | 0.484666 | top5_return_min_by_fold <= 0，缺少合理解释; worst_fold_rank_ic 明显低于当前 sl20 |
| topk30_gamma5_0 | 0.455201 | top5_return_min_by_fold <= 0，缺少合理解释; worst_fold_rank_ic 明显低于当前 sl20 |
| topk10_gamma3_0 | 0.442839 | top5_return_min_by_fold <= 0，缺少合理解释 |
| topk10_gamma5_0 | 0.424975 | top5_return_min_by_fold <= 0，缺少合理解释; worst_fold_rank_ic 明显低于当前 sl20 |
| topk20_gamma2_0 | 0.419357 | top5_return_min_by_fold <= 0，缺少合理解释 |
| v4_medium_lstm_sl40 | 0.418038 | top5_return_min_by_fold <= 0，缺少合理解释; max_drawdown 明显恶化 |
| topk10_gamma2_0 | 0.325392 | top5_return_min_by_fold <= 0，缺少合理解释 |
| sl60 | 0.316373 | top5_return_min_by_fold <= 0，缺少合理解释 |
- 若 aggressive 与 robust 结论冲突，应保留双配置，并在最终提交前按比赛冲分或稳定策略目标手动选择。
