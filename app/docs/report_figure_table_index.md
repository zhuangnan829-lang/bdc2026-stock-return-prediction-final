# 报告图表索引

本文档用于说明“图/表名称、生成脚本、对应产物、支撑结论”的对应关系，便于报告撰写、答辩定位与后续检查。

## 1. 图像索引

| 图名 | 生成脚本 | 产物文件 | 主要支撑结论 |
|---|---|---|---|
| 正式模型对比图 | `app/code/src/build_report_supplement_assets.py` | `app/docs/report_supplement_assets/formal_model_comparison_chart.png` | `LSTM sl20` 是当前综合最优正式候选。 |
| 市场阶段表现图 | `app/code/src/build_report_supplement_assets.py` | `app/docs/report_supplement_assets/market_regime_analysis_chart.png` | 模型在不同市场阶段的表现存在差异。 |
| Demo 主流程图 | `app/code/src/build_report_supplement_assets.py` | `app/docs/report_supplement_assets/demo_flowchart.png` | 项目存在从数据到提交的完整流程闭环。 |
| Demo 关键命令图 | `app/code/src/build_report_supplement_assets.py` | `app/docs/report_supplement_assets/demo_key_commands.png` | 研究链路、正式提交链路和 Docker 链路是明确分离的。 |
| Demo 结果文件说明图 | `app/code/src/build_report_supplement_assets.py` | `app/docs/report_supplement_assets/demo_result_files.png` | 关键结果文件位置清晰，便于检查。 |
| 图像插入位置说明 | `app/code/src/build_report_supplement_assets.py` | `app/docs/report_supplement_assets/report_image_insert_guide.png` | 辅助报告排版，不直接支撑模型结论。 |
| 标签分布图 | `app/code/src/generate_midterm_report_figures.py` | `app/docs/figures/midterm/fig1_label_distribution.png` | 说明训练标签分布与样本特征。 |
| Walk-forward RankIC 图 | `app/code/src/generate_midterm_report_figures.py` | `app/docs/figures/midterm/fig2_fold_rankic.png` | 说明主线模型在分折上的排序表现。 |
| 回测净值曲线图 | `app/code/src/generate_midterm_report_figures.py` | `app/docs/figures/midterm/fig3_equity_curve.png` | 说明正式方案累计收益走势。 |
| 回撤曲线图 | `app/code/src/generate_midterm_report_figures.py` | `app/docs/figures/midterm/fig4_drawdown_curve.png` | 说明正式方案风险暴露与最大回撤。 |
| 短期消融图 | `app/code/src/generate_midterm_report_figures.py` | `app/docs/figures/midterm/fig5_short_term_ablation.png` | 说明短期特征对结果的影响。 |
| 小票诊断图 | `app/code/src/generate_midterm_report_figures.py` | `app/docs/figures/midterm/fig6_ticket_diagnostics.png` | 说明第一折错排股票的诊断结果。 |

## 2. 表格索引

| 表名 | 生成脚本/来源 | 产物文件 | 主要支撑结论 |
|---|---|---|---|
| 正式模型对比表 | `app/code/src/build_formal_model_comparison.py` | `app/model/formal_model_comparison/formal_model_comparison.csv` | 正式方案为何选 `LSTM sl20`。 |
| 正式模型对比说明表 | `app/code/src/build_formal_model_comparison.py` | `app/model/formal_model_comparison/formal_model_comparison.md` | 统一口径记录模型指标。 |
| 回测汇总表 | `app/code/src/backtest.py` | `app/model/backtest_summary.csv` | 成本后收益、Sharpe、回撤、换手表现。 |
| 回测配置比较表 | `app/code/src/backtest.py` | `app/model/backtest_config_comparison.csv` | 风险过滤与权重配置的影响。 |
| fold 阶段表现表 | `app/code/src/build_market_regime_analysis.py` | `app/model/market_regime_analysis/fold_stage_performance.csv` | 不同阶段/折次的表现差异。 |
| fold 诊断表 | `run_research_pipeline.sh` 内嵌 diagnostics | `app/model/fold_diagnostics.csv` | 说明稳定性问题主要出现在什么位置。 |
| fold 日诊断表 | 同上 | `app/model/fold_daily_diagnostics.csv` | 支撑日度层面的诊断细节。 |
| 第 1 折预测明细 | 同上 | `app/model/fold_1_predictions.csv` | 支撑错排股票分析与 Demo 诊断页。 |
| 短期诊断摘要表 | 诊断分析结果沉淀 | `app/model/fold1_short_term_ticket_summary.csv` | 支撑重点股票摘要。 |
| 逐日诊断明细表 | 诊断分析结果沉淀 | `app/model/fold1_short_term_ticket_diagnostics.csv` | 支撑个股逐日解释。 |
| report 补充表集合 | `app/code/src/build_report_supplement_assets.py` | `app/docs/report_supplement_tables.md` | 报告中可直接复制引用的表格。 |

## 3. Demo 页面对应关系

| Demo 页面模块 | 主要读取文件 | 用途 |
|---|---|---|
| 项目简介与默认方案 | `app/model/best_config.json` | 展示正式默认方案快照。 |
| 模型对比 | `app/model/formal_model_comparison/formal_model_comparison.csv`、`app/docs/report_supplement_assets/formal_model_comparison_chart.png` | 展示为何选择当前主线。 |
| Walk-forward 分折结果 | `app/model/market_regime_analysis/fold_stage_performance.csv`、`app/docs/figures/midterm/fig2_fold_rankic.png` | 展示排序稳定性与阶段差异。 |
| 回测表现 | `app/model/backtest_summary.csv`、`fig3_equity_curve.png`、`fig4_drawdown_curve.png` | 展示收益与风险。 |
| 错排诊断 | `app/model/fold_1_predictions.csv`、`app/model/fold1_short_term_ticket_summary.csv`、`app/model/fold1_short_term_ticket_diagnostics.csv`、`fig6_ticket_diagnostics.png` | 展示可解释性与失败分析。 |

## 4. 报告引用建议

如果报告篇幅有限，优先放这几张图/表：

1. `formal_model_comparison_chart.png`
2. `formal_model_comparison.csv`
3. `fold_stage_performance.csv`
4. `fig3_equity_curve.png`
5. `fig4_drawdown_curve.png`
6. `fold1_short_term_ticket_diagnostics.csv`

这样能覆盖：

- 模型选择
- 稳定性
- 风险收益
- 失败案例分析
