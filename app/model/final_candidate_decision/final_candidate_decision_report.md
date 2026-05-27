# Final Candidate Decision Report

本报告用于人工确认前的最终候选决策；不会覆盖 `app/output/result.csv`，也不会覆盖 `app/model/default_submission_config.json`。

## 结论

- 默认建议：当前默认结果已经可以作为权威提交。
- HV rerank 已按人工确认同步为默认提交。
- aggressive 只适合比赛冲分目标，当前不直接提交。
- robust 只适合稳定/低换手目标，当前不作为默认提交。
- 是否同步默认配置：已同步 HV rerank。

## 最终候选决策表

| candidate | role | submit_decision | reason | cannot_submit_reason | sync_default_config |
| --- | --- | --- | --- | --- | --- |
| current_default_sl20 | 当前权威默认提交，已同步 HV rerank | 建议保留为默认提交 | 已按人工确认切换为 HV rerank；它仍使用 LSTM sl20 主线，只增加高波动阶段轻重排。 | 无。当前就是默认正式结果。 | 不需要 |
| aggressive | 比赛冲分配置建议 | 不建议直接提交 | 它追求收益和单切片，保留 pred 权重、宽候选池和较高换手，适合冲分思路。 | 当前没有作为独立正式候选重新生成 result 并通过提交级复核；且高换手/集中风险更高。 | 只有人工明确选择比赛冲分时才同步 |
| robust | 稳定/低换手配置建议 | 不建议作为本轮默认提交 | 它降低换手、单票集中和纯 pred 依赖，适合高波动震荡或保守展示。 | 收益弹性明显低于 aggressive/默认主线，且当前未作为独立正式候选生成提交文件。 | 只有目标切换为稳定策略时才同步 |
| hv_rerank | 最值得人工确认的增强候选 | 已同步为当前默认提交 | 它不替换 sl20 模型，只在高波动 regime 触发 close_position_20d 轻微重排；本轮最新数据确实触发高波动状态。 | 无。已经人工确认并同步为当前默认。 | 已同步 |

## 证据表

| candidate | evidence | manual_action |
| --- | --- | --- |
| current_default_sl20 | validator 通过, rows=5, weight_sum=0.900000；sl=20, cs=180, sort=risk_adjusted, weight=pred, cap=0.18, mt=1.0, rerank=is_high_volatility/close_position_20d:-0.05。 | 若不想承担候选切换风险，直接保留当前 result.csv。 |
| aggressive | sl=20, cs=180, sort=risk_adjusted, weight=pred, cap=0.2, mt=1.0, blend_alpha=1.0；最终选择器将其定位为冲分配置。 | 若要选它，先单独生成 aggressive result，再跑 validator、pre-submit 和与当前结果对比。 |
| robust | turnover_stress: return=0.653410, sharpe=4.053580, max_drawdown=-0.048990, avg_turnover=0.500000。 sl=20, cs=180, sort=risk_adjusted, weight=pred_equal_blend, cap=0.2, mt=0.5, blend_alpha=0.5。 | 若目标变成低回撤/低换手，先生成 robust result，再做提交级验证。 |
| hv_rerank | validator 通过, rows=5, weight_sum=0.900000；regime_rerank: delta_return=0.106056, delta_sharpe=0.187586, delta_max_drawdown=0.002645, delta_high_vol=0.002761。 相对当前结果保留 4 只，新增 600115，移除 601877，切换换手 0.360000。 | 当前默认 result.csv 已经是 HV rerank 结果；后续只需做最终打包前检查。 |

## 文件

- decision csv: `D:\Desktop\股票分析预测代码\app\model\final_candidate_decision\final_candidate_decision_table.csv`
- current result: `D:\Desktop\股票分析预测代码\app\output\result.csv`
- HV rerank result: `D:\Desktop\股票分析预测代码\app\model\hv_rerank_submission_candidate\result_hv_rerank.csv`
- HV rerank comparison: `D:\Desktop\股票分析预测代码\app\model\hv_rerank_submission_candidate\result_hv_rerank_vs_current.csv`
- full validation report: `D:\Desktop\股票分析预测代码\app\model\full_optimization_validation_report.md`
- final config selection report: `D:\Desktop\股票分析预测代码\app\model\configs\final_config_selection_report.md`

## 手动确认提示

如果最终选择当前默认结果：不需要同步配置。

HV rerank 已经同步为当前默认结果；当前正式 `result.csv` 包含 `600115`，不再包含 `601877`。

如果最终选择 aggressive 或 robust：请先生成对应独立提交文件并通过 validator / pre-submit，再决定是否同步默认配置。
