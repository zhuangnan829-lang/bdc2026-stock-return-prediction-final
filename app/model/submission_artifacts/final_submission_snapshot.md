# 最终提交快照

## 当前提交变体

- 变体名称：`aggressive_score_submission`
- 用途：追求可见单切片分数的满仓候选方案
- 可见 case-slice score：`0.077484`
- 结果来源：`D:/Desktop/股票分析预测代码/app/model/aggressive_score_submission_candidate/result_aggressive_score.csv`

## 包内保留的默认模型配置

- 配置名称：`lstm_sl20_base_alpha_v3_rs_crowding_mini4__hv_close_position_rerank`
- 配置状态：`frozen_best_config`
- 主模型：`LSTM`
- 特征集：`base_alpha_v3_rs_crowding_mini4`
- 特征数：`20`
- 训练目标：`cross_section_rank`
- 序列窗口长度：`20`
- 排序策略：`risk_adjusted`
- 权重策略：`pred`
- 选股数量：`5`
- 候选池大小：`180`
- `max_volatility_20d_pct = 0.86`
- `max_volatility_5d_pct = 1.00`
- `turnover_rate_lower_pct = 0.03`
- `turnover_rate_upper_pct = 0.97`
- `turnover_ratio_upper_pct = 0.95`
- `risk_penalty_weight = -0.30`
- `max_turnover = 1.00`
- `transaction_cost = 0.001`
- `AUTO_USE_PREVIOUS_RESULT = 0`

配置来源：
- 权威配置源：[default_submission_config.json](D:/Desktop/股票分析预测代码/app/model/default_submission_config.json:1)
- 派生同步文件：[best_config.json](D:/Desktop/股票分析预测代码/app/model/best_config.json:1)
- [model_meta.json](D:/Desktop/股票分析预测代码/app/model/model_meta.json:1)

## 当前提交文件

- [result.csv](D:/Desktop/股票分析预测代码/app/output/result.csv:1)

当前持仓如下：

| stock_id | weight |
|---|---:|
| `000792` | `0.200000` |
| `600233` | `0.200000` |
| `601669` | `0.200000` |
| `600930` | `0.200000` |
| `002463` | `0.200000` |

当前 `result.csv` 权重和为：`1.000000`

说明：
- 当前 `result.csv` 以 aggressive 变体同步后的满仓结果为准
- 默认 LSTM 冻结模型和配置仍保留在包内，用于复现推理链路
- 当前正式提交路径默认不自动复用上一版 `result.csv`

## 当前默认方案本地回测指标

- 成本后累计收益：`1.171246`
- 成本后 Sharpe：`4.019488`
- 成本后最大回撤：`-0.090067`
- 平均换手率：`0.956671`
- walk-forward `rank_ic_mean`：`0.027982`
- walk-forward `top5_mean_return_mean`：`0.007502`

指标来源：
- [backtest_summary.csv](D:/Desktop/股票分析预测代码/app/model/default_profile_backtest/backtest_summary.csv:1)

## 与压缩包程序的当前对比结论

- 我方当前单切片分数：`0.077484`
- 压缩包当前可见分数：`0.025179`
- 压缩包公开最佳分数：`0.037838`

这说明：
- 当前 aggressive 提交结果超过参考当前可见输出和参考记录最好分数

对应文件：
- [case_program_comparison_summary.csv](D:/Desktop/股票分析预测代码/app/model/case_program_comparison/case_program_comparison_summary.csv:1)
- [case_program_comparison_report.md](D:/Desktop/股票分析预测代码/app/model/case_program_comparison/case_program_comparison_report.md:1)
- [latest_score_compare.md](D:/Desktop/股票分析预测代码/app/model/case_comparison/latest_score_compare.md:1)

## 提交校验状态

- 正式默认脚本参数与配置快照一致
- `result.csv` 符合提交格式要求
- `pre_submit_check.py` 一致性检查应通过
