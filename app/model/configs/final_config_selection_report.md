# Final Submission Config Selection Report

This report is advisory only. It does not overwrite `app/model/default_submission_config.json`.

## 1. Aggressive 配置是谁，为什么

- 建议 aggressive 配置: `aggressive_lstm_sl20_alpha_v3_rs_crowding_mini4_cs180_rp-30_pred_mt100_cap20`: model=lstm, feature_set=base_alpha_v3_rs_crowding_mini4, sl=20, cs=180, sort=risk_adjusted, weight=pred, blend_alpha=1.0, cap=0.2, rp=-0.3, max_turnover=1.0, tc=0.001
- 原因: 该配置使用 `pred` 权重、`primary_candidate_size=180` 宽候选池、`max_turnover=1.0`，目标是单切片分数和累计收益。
- weight_cap cap=0.20: single_slice_score=0.030099, cost_after_return=1.165010, sharpe=4.002179, max_drawdown=-0.090852, avg_turnover=0.956291, max_single_weight=0.200000, max_single_contribution_ratio=0.882825
- weight_cap cap=none: single_slice_score=0.029985, cost_after_return=1.171246, max_drawdown=-0.090067, avg_turnover=0.956671, max_single_contribution_ratio=0.881393
- leaderboard top cumulative_return_after_cost: candidate_label=topk10_gamma5_0, cumulative_return_after_cost=1.718410, sharpe_after_cost=4.614816, max_drawdown_after_cost=-0.070357, avg_turnover=0.957888, slice_score=n/a

## 2. Robust 配置是谁，为什么

- 建议 robust 配置: `robust_lstm_sl20_alpha_v3_rs_crowding_mini4_cs180_rp-30_blend05_mt050_cap20`: model=lstm, feature_set=base_alpha_v3_rs_crowding_mini4, sl=20, cs=180, sort=risk_adjusted, weight=pred_equal_blend, blend_alpha=0.5, cap=0.2, rp=-0.3, max_turnover=0.5, tc=0.001
- 原因: 该配置使用 `blend_0.5` 风格的 pred/equal 混合、`max_single_weight=0.20`、`max_turnover=0.50`，更适合高波动震荡阶段。
- turnover stress mt050_tc0010_blend_0.5_cap0.20: cost_after_return=0.653410, sharpe=4.053580, max_drawdown=-0.048990, avg_turnover=0.500000, robust_score=1.271209, win_rate=0.683333
- weight_blend alpha=0.5 cap=0.20: single_slice_score=0.030099, cost_after_return=1.165010, sharpe=4.002179, max_drawdown=-0.090852, avg_turnover=0.956291, max_single_contribution_ratio=0.882825
- weight_cap cap=0.18: single_slice_score=0.027089, cost_after_return=1.085936, sharpe=3.977455, max_drawdown=-0.087064, avg_turnover=0.931137, max_single_contribution_ratio=0.876081

## 2b. HV rerank 候选是谁，为什么

- 建议 HV rerank 候选: `lstm_sl20_base_alpha_v3_rs_crowding_mini4__hv_close_position_rerank`: model=lstm, feature_set=base_alpha_v3_rs_crowding_mini4, sl=20, cs=180, sort=risk_adjusted, weight=pred, blend_alpha=1.0, cap=0.18, rp=-0.3, max_turnover=1.0, tc=0.001, regime_rerank=is_high_volatility/close_position_20d:-0.05
- 原因: 该配置不替换 LSTM sl20 主线，只在 `is_high_volatility=1` 时对 `close_position_20d` 施加 -0.05 轻微重排惩罚，用于处理高波动阶段的误排样本。
- regime_rerank baseline: cost_after_return=1.085936, sharpe=3.977455, max_drawdown=-0.087064, avg_turnover=0.931137, selected_top5_return_mean=0.016264, high_volatility_selected_top5_return=0.018838, poor_false_positives=128.000000
- regime_rerank hv_close_position_20d_m005: cost_after_return=1.191992, sharpe=4.165041, max_drawdown=-0.084420, avg_turnover=0.929450, selected_top5_return_mean=0.017184, high_volatility_selected_top5_return=0.021600, delta_cost_after_return=0.106056, delta_sharpe=0.187586, delta_max_drawdown=0.002645, delta_poor_false_positives=-3.000000
- regime_rerank best cost_after_return: profile_name=hvrange_close_position_20d_m005, cost_after_return=1.195231, sharpe=4.155355, max_drawdown=-0.087064, avg_turnover=0.929427, delta_cost_after_return=0.109295

## 3. 最终默认建议用哪个

- 默认建议: 暂时保留当前 LSTM sl20 默认主线；新增 `lstm_sl20_base_alpha_v3_rs_crowding_mini4__hv_close_position_rerank` 作为优先复核候选。
- 理由: aggressive/robust 仍然服务于不同提交目标；HV rerank 在现有回测中同时改善收益、Sharpe、高波动 Top5 和误报数量，但它是后处理重排，需要再经过完整提交文件验证后才能同步默认配置。
- 注意: 本脚本不会自动同步默认配置，需要人工确认后再决定是否执行同步。

## 4. 如果目标是比赛冲分，选哪个

- 选择 `aggressive_lstm_sl20_alpha_v3_rs_crowding_mini4_cs180_rp-30_pred_mt100_cap20`。
- 采用证据: cap=0.20 保留最高单切片/累计收益附近的表现，`pred` 权重和 full turnover 不主动压制冲分能力。
- 放弃 robust 的原因: robust 主动限制换手并混合权重，适合稳健但会牺牲一部分收益弹性。
- 可加测 `lstm_sl20_base_alpha_v3_rs_crowding_mini4__hv_close_position_rerank`: 它保留 pred 权重和宽候选池，同时只在高波动阶段轻微重排；若最终提交文件验证通过，可作为冲分增强候选。

## 5. 如果目标是稳定策略，选哪个

- 选择 `robust_lstm_sl20_alpha_v3_rs_crowding_mini4_cs180_rp-30_blend05_mt050_cap20`。
- 采用证据: turnover stress 中 `mt050_tc0010_blend_0.5_cap0.20` 在低换手约束下保留较高 robust_score；cap=0.20 控制单票权重，blend_0.5 降低纯 pred 权重依赖。
- 放弃 aggressive 的原因: aggressive 接受更高换手和集中度，遇到高波动震荡阶段时执行和回撤风险更高。
- 若目标是“不明显降收益的稳定增强”，优先观察 `lstm_sl20_base_alpha_v3_rs_crowding_mini4__hv_close_position_rerank`；若目标是强制低换手，则仍选 robust。

## 6. 采用/放弃每个配置的证据

### `aggressive_lstm_sl20_alpha_v3_rs_crowding_mini4_cs180_rp-30_pred_mt100_cap20`

- 采用: 宽候选池 `cs180`、`pred` 权重、`max_turnover=1.0` 与冲分目标一致。
- 采用: weight_cap 中 cap=0.20 的 `single_slice_score` 和 `cost_after_return` 保持强势。
- 放弃作为稳健默认: `avg_turnover` 和 `max_single_contribution_ratio` 偏高，风险暴露更集中。

### `robust_lstm_sl20_alpha_v3_rs_crowding_mini4_cs180_rp-30_blend05_mt050_cap20`

- 采用: `blend_0.5`、`max_single_weight=0.20`、`max_turnover=0.50` 同时处理权重集中和换手压力。
- 采用: turnover stress 的低换手配置给出可接受收益与 Sharpe，适合震荡阶段。
- 放弃作为比赛冲分默认: 累计收益上限低于 aggressive 路线，可能错过单切片最优权重。

### `lstm_sl20_base_alpha_v3_rs_crowding_mini4__hv_close_position_rerank`

- 采用: `hv_close_position_20d_m005` 相对 baseline 提高 `cost_after_return`、`sharpe`，并小幅改善 `max_drawdown` 和高波动阶段 Top5 收益。
- 采用: 仅在高波动 regime 触发，避免把防守惩罚扩散到全部市场阶段。
- 放弃直接同步默认: 这是重排后处理候选，还需要重新生成提交文件并通过 `result_validator.py`、`pre_submit_check.py` 和完整验证流水线。

## Stability 参考

- target stability row: n/a
- best stability_score row: model=lstm, feature_set=base_alpha_v3_rs_crowding_mini4, sequence_length=20.000000, rank_ic_mean=0.027982, worst_fold_rank_ic=-0.033492, top5_return_mean=0.007502, stability_score=-0.037332

## Input Files

- experiment_leaderboard.csv: `D:\Desktop\股票分析预测代码\app\model\experiment_leaderboard.csv`
- turnover_stress_summary.csv: `D:\Desktop\股票分析预测代码\app\model\turnover_stress_test\turnover_stress_summary.csv`
- weight_cap_summary.csv: `D:\Desktop\股票分析预测代码\app\model\weight_cap_search\weight_cap_summary.csv`
- weight_blend_summary.csv: `D:\Desktop\股票分析预测代码\app\model\weight_blend_search\weight_blend_summary.csv`
- stability_summary.csv: `D:\Desktop\股票分析预测代码\app\model\stability_eval\stability_summary.csv`
- regime_rerank_switch_summary.csv: `D:\Desktop\股票分析预测代码\app\model\regime_rerank_switch\regime_rerank_switch_summary.csv`
- aggressive config: `D:\Desktop\股票分析预测代码\app\model\configs\submission_aggressive.json`
- robust config: `D:\Desktop\股票分析预测代码\app\model\configs\submission_robust.json`
- hv rerank config: `D:\Desktop\股票分析预测代码\app\model\configs\submission_hv_rerank_candidate.json`

## Manual Confirmation Required

请手动确认是否将建议配置同步到 `app/model/default_submission_config.json`；本报告和脚本不会自动覆盖默认配置。
