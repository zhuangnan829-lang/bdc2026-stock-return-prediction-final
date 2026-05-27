# `app/model` 目录用途说明

本目录同时保存了：

1. 正式提交所需的冻结产物
2. 主线实验结果
3. 基线对比结果
4. 消融、搜索与历史研究档案

因为内容较多，建议按下面方式理解。

## 1. 正式交付产物

这部分是**正式提交链路**最重要的内容，优先关注：

- `best_config.json`
- `default_submission_config.json`
- `model_meta.json`
- `lstm_model.pt`
- `final_submission_snapshot.md`
- `submission_artifacts/`

用途：

- 说明当前冻结默认方案是什么
- 支撑 `run_submission.sh` / `freeze_submission.sh`
- 作为答辩时“正式口径”的依据

## 2. 当前主线结果

这部分是当前正式主线 `LSTM sl20` 的核心研究结果：

- `walk_forward_predictions.csv`
- `walk_forward_metrics.csv`
- `backtest_summary.csv`
- `backtest_report.md`
- `fold_diagnostics.csv`
- `fold_daily_diagnostics.csv`
- `fold_1_predictions.csv`
- `fold1_short_term_ticket_summary.csv`
- `fold1_short_term_ticket_diagnostics.csv`

用途：

- 说明主线模型的排序能力
- 说明成本后回测表现
- 说明第一折与错排诊断

## 3. 正式模型对比与阶段分析

这部分用于支撑“为什么最终选择当前主线”：

- `formal_model_comparison/`
- `market_regime_analysis/`
- `model_comparison/`

用途：

- 展示主线与基线/备选模型之间的差异
- 展示不同市场阶段下的表现差异

## 4. 基线与备选模型目录

这部分不是正式提交必需，但用于研究对照：

- `baseline_lightgbm_same_protocol/`
- `baseline_linear_same_protocol/`
- `xgboost_baseline/`
- `transformer_baseline/`

用途：

- 支撑机器学习、线性模型和 Transformer 基线对照
- 用于答辩时解释“做过哪些备选方案”

## 5. 消融、搜索与历史实验档案

这部分主要是**研究档案**，不属于正式提交最小必需集合：

- `ablation*/`
- `alpha_*`
- `lstm_*search*`
- `top40_*`
- `trend_rerank_signal_experiment/`
- `turnover_pred_local_search/`
- `weighting_risk_joint_search/`
- `sort_weight_turnover_joint_search/`
- `report_materials/`

用途：

- 记录特征消融、参数搜索、候选方案筛选过程
- 作为报告和答辩时的补充证据

## 6. 如何快速定位

如果你是第一次进入 `app/model`，建议这样看：

1. `final_submission_snapshot.md`
2. `formal_model_comparison/`
3. `backtest_summary.csv`
4. `market_regime_analysis/`
5. `fold1_short_term_ticket_diagnostics.csv`
6. 其余历史实验目录

## 7. 最小交付集合

如果只保留“正式提交 + 答辩最关键材料”，最小集合大致是：

- `best_config.json`
- `default_submission_config.json`
- `model_meta.json`
- `lstm_model.pt`
- `final_submission_snapshot.md`
- `submission_artifacts/`
- `formal_model_comparison/`
- `backtest_summary.csv`
- `market_regime_analysis/`
- `fold_1_predictions.csv`
- `fold1_short_term_ticket_summary.csv`
- `fold1_short_term_ticket_diagnostics.csv`

更完整的实验索引见：

- [experiment_result_index.md](/d:/Desktop/股票分析预测代码/app/docs/experiment_result_index.md)
