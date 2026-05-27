Prompt 13：实现 regime-aware 融合与候选选择
请实现一个不优先改训练目标的轻量实验，用于验证 fusion/ensemble 与 regime-aware candidate selection 是否能在不伤 worst fold 的情况下提升稳健性。

背景约束：
当前证据显示，训练目标改造（例如 topk_weighted_rank、clipped_return + topk_weighted_rank）可能拉高局部收益或 Sharpe，但容易牺牲 worst_fold_rank_ic，尤其 Fold 3 与高波动阶段风险较硬。第 13 步不再优先新增训练目标或大规模重训。

优先方向：
1. fusion / ensemble：基于已有 walk-forward predictions、已有模型预测或已有候选分数做 average_rank 融合，不直接平均原始预测值。
2. regime-aware candidate selection：在低波动阶段保留 aggressive 默认排序；仅在 high_volatility 或 high_volatility_range 阶段启用轻量防守排序。
3. 已验证优先候选：high_volatility 日期启用 close_position_20d，rerank_signal_weight=-0.05；低波动日期不启用 rerank。

新增或修改文件建议：
app/code/src/regime_aware_fusion.py
app/code/src/regime_rerank_switch.py（如已有则复用/扩展）
app/model/configs/submission_robust_regime_rerank_candidate.json（候选配置，不覆盖 default）

输入：
--pred_paths 一个或多个预测文件
--feature_path app/temp/train_features.csv
--regime_path app/model/market_regime_analysis/daily_market_regimes.csv
--output_dir app/model/regime_aware_fusion/
--risk_signals close_position_20d reversal_risk_score
--risk_weights -0.05
--regime_flags is_high_volatility is_high_volatility_range

实验配置：
1. baseline：当前 aggressive 主线，不加 regime rerank。
2. global_close_position_20d_m005：全局 close_position_20d -0.05，仅作对照。
3. hv_close_position_20d_m005：仅 high_volatility 日期启用 close_position_20d -0.05。
4. hvrange_close_position_20d_m005：仅 high_volatility_range 日期启用 close_position_20d -0.05。
5. hv_reversal_risk_score_m005：仅 high_volatility 日期启用 reversal_risk_score -0.05。
6. 可选 average_rank fusion：baseline rank 与 defensive rank 按 regime 切换或加权融合。

输出目录：
app/model/regime_aware_fusion/

输出：
regime_aware_fusion_summary.csv
regime_aware_fusion_report.md
每个候选 profile 的 backtest_summary.csv、backtest_daily.csv、result.csv
如形成候选配置，输出到 app/model/configs/，但不得覆盖 default_submission_config.json。

指标：
rank_ic_mean
worst_fold_rank_ic
fold1_rank_ic
fold3_rank_ic
negative_day_rank_ic_ratio
top5_return_mean
high_volatility_top5_return
high_volatility_range_top5_return
cost_after_return
Sharpe
max_drawdown
avg_turnover
single_slice_score
false_positives
poor_false_positives

报告必须回答：
1. regime-aware 防守是否改善 Fold 3？
2. high_volatility / high_volatility_range 阶段是否改善？
3. 低波动 aggressive 是否被保留，没有被无谓削弱？
4. close_position_20d -0.05 是否优于 reversal_risk_score -0.05？
5. 是否值得生成 robust 候选配置？
6. 是否仍建议保留 aggressive / robust 双配置？

采用规则：
只有当 regime-aware 配置至少满足以下条件，才进入 robust 候选：
- Fold 3 或 high_volatility Top5 收益改善；
- worst_fold_rank_ic 不明显差于当前主线；
- cost_after_return / Sharpe 不明显恶化；
- poor_false_positives 不增加；
- result_validator 通过；
- 可通过候选配置和 cli.py predict 稳定复现。

禁止事项：
- 不新增默认训练目标；
- 不覆盖 default_submission_config.json；
- 不把单切片收益提升作为唯一采用依据；
- 不因为单个 seed 或单个 fold 偶然提升就进入主线。
