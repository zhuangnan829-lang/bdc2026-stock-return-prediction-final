# 实验结果索引

本文档承接从根 README 移出的研发记录，用于把“实验脚本、结果文件、结论用途”串起来。正式使用入口请看仓库根目录 `README.md`；这里主要服务于论文、答辩、复盘和结果追溯。

## 1. 正式主线与模型对比

| 实验主题 | 主要脚本 | 主要结果文件 | 用途/支撑结论 |
|---|---|---|---|
| 正式模型对比 | `app/code/src/build_formal_model_comparison.py` | `app/model/formal_model_comparison/formal_model_comparison.csv`、`app/model/formal_model_comparison/formal_model_comparison.md` | 支撑“为什么正式方案选择 LSTM sl20”。 |
| 综合模型比较 | `app/code/src/run_model_comparison.py` | `app/model/model_comparison/model_comparison_summary.csv`、`app/model/model_comparison/model_comparison_report.md` | 说明除正式候选外还评估过哪些模型。 |
| 树模型补充对比 | `scripts/compare_tree_models.py` | `app/model/model_comparison/tree_model_comparison.csv`、`app/model/model_comparison/tree_model_comparison.md` | 对 LightGBM、XGBoost 等树模型结果做补充归档。 |
| 最终冻结方案 | `app/code/src/sync_submission_config.py`、`app/freeze_submission.sh` | `app/model/final_submission_snapshot.md`、`app/model/submission_artifacts/` | 说明当前正式默认配置是什么。 |

当前正式结论：`LSTM sl20 + base_alpha_v3_rs_crowding_mini4 + risk_adjusted + pred` 是提交主线；LightGBM、XGBoost、Linear Regression 和 Momentum 作为对照基线保留。

## 2. 训练、walk-forward 与预测产物

| 实验主题 | 主要脚本 | 主要结果文件 | 用途/支撑结论 |
|---|---|---|---|
| LSTM 训练 | `app/train.sh`、`app/code/src/train_lstm.py` | `app/model/lstm_model.pt`、`app/model/model_meta.json` | 支撑主线模型已完成训练。 |
| walk-forward 验证 | `app/code/src/train_lstm.py` | `app/model/walk_forward_predictions.csv`、`app/model/walk_forward_metrics.csv` | 评估主线模型排序能力。 |
| 冻结推理 | `app/test.sh`、`app/code/src/test_lstm.py` | `app/output/result.csv`、`app/output/predict_scores.csv`、`app/output/debug_candidates.csv` | 支撑正式提交结果生成链路。 |
| 结果校验 | `app/code/src/result_validator.py`、`app/code/src/pre_submit_check.py` | 终端校验结果、提交前检查结果 | 校验 `result.csv` 格式、数量和必要文件。 |

## 3. 本地回测与风险收益分析

| 实验主题 | 主要脚本 | 主要结果文件 | 用途/支撑结论 |
|---|---|---|---|
| 默认回测与配置比较 | `app/code/src/backtest.py` | `app/model/backtest_summary.csv`、`app/model/backtest_report.md`、`app/model/backtest_config_comparison.csv` | 支撑正式方案收益、Sharpe、回撤表现。 |
| 回测净值与回撤图 | `app/code/src/generate_midterm_report_figures.py` | `app/docs/figures/midterm/fig3_equity_curve.png`、`app/docs/figures/midterm/fig4_drawdown_curve.png` | 用于报告和答辩展示。 |
| 日频持仓与回测明细 | `app/code/src/backtest.py` | `app/model/backtest_daily_results.csv`、`app/model/backtest_holdings.csv` | 支撑更细粒度的组合回放分析。 |
| 回测压力测试 | `app/code/src/backtest_stress_test.py` | `app/model/backtest_stress_test/` | 观察交易成本、换手、权重约束下的稳健性。 |

## 4. 稳定性、分折与市场阶段分析

| 实验主题 | 主要脚本 | 主要结果文件 | 用途/支撑结论 |
|---|---|---|---|
| fold 诊断 | `app/run_research_pipeline.sh` 内置 diagnostics 流程 | `app/model/fold_diagnostics.csv`、`app/model/fold_daily_diagnostics.csv`、`app/model/fold_1_predictions.csv` | 分析分折稳定性与第 1 折表现退化。 |
| 排名稳定性评估 | `app/code/src/evaluate_rank_stability.py`、`app/code/src/run_stability_suite.py` | `app/model/stability_eval/` | 观察不同参数与样本切分下的排名稳定性。 |
| 市场阶段分析 | `app/code/src/build_market_regime_analysis.py`、`app/code/src/evaluate_by_market_regime.py` | `app/model/market_regime_analysis/` | 支撑“不同市场阶段下模型表现不同”。 |
| 报告补充图资产 | `app/code/src/build_report_supplement_assets.py` | `app/docs/report_supplement_assets/market_regime_analysis_chart.png` | 用于中期报告和答辩图示。 |

## 5. 特征消融与解释性分析

| 实验主题 | 主要脚本/目录 | 主要结果文件 | 用途/支撑结论 |
|---|---|---|---|
| 特征消融汇总 | `scripts/run_ablation_study.py`、`app/model/ablation/` | `app/model/ablation/ablation_summary.csv`、`app/model/ablation/ablation_report.md` | 支撑当前特征集如何收敛。 |
| Alpha v3 消融 | `app/code/src/run_alpha_v3_ablation.py` | `app/model/alpha_v3_ablation/alpha_v3_ablation_summary.csv`、`app/model/alpha_v3_ablation/alpha_v3_ablation_report.md` | 解释 alpha v3 特征组合贡献。 |
| Alpha v4 micro 实验 | `app/code/src/run_alpha_v4_micro_experiment.py`、`app/code/src/run_alpha_v4_micro_ablation.py` | `app/model/alpha_v4_micro_ablation/`、`app/output/result_alpha_v4_micro.csv` | 记录候选新特征未替代正式方案的证据。 |
| 个股误排诊断 | `app/code/src/diagnose_misranked_samples.py` | `app/model/misrank_diagnostics/`、`app/docs/figures/midterm/fig6_ticket_diagnostics.png` | 解释部分样本短期因子导致的错排。 |
| 特征漂移监控 | `app/code/src/analyze_feature_drift.py`、`app/code/src/feature_importance_report.py` | `app/model/*/feature_drift_monitoring/`、`app/model/*/feature_importance.csv` | 识别特征稳定性和重要性变化。 |

## 6. 基线与备选模型实验

| 实验主题 | 主要结果目录 | 主要结果文件 | 用途/支撑结论 |
|---|---|---|---|
| LightGBM 基线 | `app/model/baseline_lightgbm_same_protocol/` | `model_meta.json`、`walk_forward_predictions.csv` | 机器学习基线对照。 |
| XGBoost 基线 | `app/model/xgboost_baseline/` | `model_meta.json`、`walk_forward_predictions.csv`、`walk_forward_metrics.csv` | 树模型基线对照。 |
| 线性回归基线 | `app/model/baseline_linear_same_protocol/` | 相关 baseline 结果文件 | 线性模型对照。 |
| Transformer / Transformer Lite | `app/model/transformer_lite_sl60/`、相关 `result_transformer.csv` | `walk_forward_metrics.csv`、`walk_forward_predictions.csv`、对比 CSV | 记录做过 Transformer 类方案，但未纳入正式方案。 |
| Momentum 规则基线 | 模型对比结果目录 | `app/model/model_comparison/model_comparison_summary.csv` | 规则基线参照。 |

## 7. 候选池、排序、权重与换手搜索

| 实验主题 | 主要脚本 | 主要结果目录/文件 | 用途/支撑结论 |
|---|---|---|---|
| 候选池搜索 | `app/code/src/search_candidate_pool.py`、`scripts/search_inference_grid.py` | `app/model/candidate_pool_search/`、`app/model/candidate_frontend_refine_experiment/` | 选择候选池大小与前端过滤参数。 |
| 排序与权重联合搜索 | `scripts/search_sort_weight_turnover_joint.py`、`scripts/search_weighting_risk_joint.py` | `app/model/weight_blend_search/`、`app/model/weight_cap_search/` | 支撑排序、权重、风险惩罚、换手约束设置。 |
| Top-k 目标搜索 | `app/code/src/train_lstm_topk_weighted.py`、相关 top-k 搜索脚本 | `app/model/topk_objective_search/` | 评估 Top5 目标与排序目标差异。 |
| 换手压力测试 | `app/code/src/turnover_stress_test.py`、`scripts/search_turnover_pred_local.py` | `app/model/turnover_stress_test/` | 评估换手限制对收益和稳定性的影响。 |

## 8. Docker、交付与一致性验证

| 验证主题 | 主要脚本 | 主要结果文件 | 用途/支撑结论 |
|---|---|---|---|
| Docker 离线彩排 | `Dockerfile`、`docker-compose.yml`、`app/docker_rehearsal.sh` | `test/output/result.csv`、`test/output/predict_scores.csv`、`test/output/debug_candidates.csv` | 验证容器内可生成正式输出。 |
| 本机与 Docker 对比 | `app/code/src/compare_local_docker_result.py` | `app/model/docker_consistency_check.md` | 记录本机和容器结果一致性。 |
| 必要文件检查 | `app/code/src/check_required_files.py` | 终端检查结果、测试用例结果 | 验证交付包关键文件齐备。 |
| 配置一致性检查 | `app/code/src/compare_config_consistency.py` | 终端检查结果 | 确保 best/default/meta 配置口径一致。 |

## 9. Demo 与展示材料

| 展示主题 | 主要脚本 | 主要结果文件 | 用途/支撑结论 |
|---|---|---|---|
| Streamlit Demo | `app/demo/streamlit_app.py` | 读取 `app/model/` 与 `app/docs/` 下现有结果文件 | 支撑答辩现场展示。 |
| Demo 流程图与说明图 | `app/code/src/build_report_supplement_assets.py` | `app/docs/report_supplement_assets/demo_flowchart.png`、`demo_key_commands.png`、`demo_result_files.png` | 说明项目是完整流程而不是脚本堆叠。 |
| 3 分钟展示主线 | 手工文档 | `app/docs/demo_3min_main_flow.md` | 答辩时快速串联“入口、结果、验证、解释”。 |

## 10. 建议优先引用的证据文件

如果只允许快速展示少量文件，建议按下面顺序引用：

1. `app/model/final_submission_snapshot.md`
2. `app/output/result.csv`
3. `app/model/formal_model_comparison/formal_model_comparison.csv`
4. `app/model/backtest_summary.csv`
5. `app/model/market_regime_analysis/fold_stage_performance.csv`
6. `app/model/fold_diagnostics.csv`
7. `app/model/fold_daily_diagnostics.csv`
8. `app/model/submission_artifacts/README.md`
9. `app/demo/streamlit_app.py`

## 11. 与 README 的边界

- 根目录 `README.md`：告诉正式使用者怎么跑、产物在哪里、怎么校验、常见问题怎么处理。
- 本文档：保存研发过程、实验证据链和结果索引。
- `app/docs/reproducibility_guide.md`：展开复现步骤。
- `app/docs/final_delivery_checklist.md`：最终交付前逐项检查。
