# 最终展示统一口径表

本文档用于统一 README、结题报告、PPT、Demo 和答辩口头表述。仓库当前没有正式 `.pptx` 文件，因此这里提供的是 PPT 制作与逐页核对的权威口径；后续制作 PPT 时应逐项照此填写，不再单独改写模型清单、默认方案或图表路径。

## 1. 一句话总口径

本项目围绕沪深300成分股历史行情，完成了“数据处理、特征工程、模型训练、横截面排序、Top-K 股票推荐、组合回测、冻结提交与 Docker 复现”的完整闭环。正式提交包为 `aggressive_score_submission`，最终 `result.csv` 使用 aggressive 满仓候选结果；包内同时保留 `LSTM sl20` 默认模型与配置，用于复现推理链路和解释模型选择依据。

## 2. PPT 模型清单口径

PPT 中模型清单应使用下表，不再扩大或缩小模型范围。

| 模型/方向 | PPT 状态写法 | 证据文件 | 展示结论 |
|---|---|---|---|
| LSTM sl20 | 正式主线/默认模型 | `app/model/final_submission_snapshot.md`、`app/model/formal_model_comparison/formal_model_comparison.csv` | 综合排序、回测和复现表现最好，是包内保留的默认模型。 |
| LightGBM | 机器学习基线 | `app/model/formal_model_comparison/formal_model_comparison.csv` | 已完成同协议对照，但综合表现弱于 LSTM。 |
| XGBoost | 树模型基线 | `app/model/xgboost_baseline/`、`app/model/formal_model_comparison/formal_model_comparison.csv` | 已完成同特征集树模型对照，不进入正式候选。 |
| Linear Regression | 线性基线 | `app/model/formal_model_comparison/formal_model_comparison.csv` | 用于证明非线性模型收益，不进入正式候选。 |
| Momentum | 规则基线 | `app/model/formal_model_comparison/formal_model_comparison.csv` | 提供简单动量策略参照，不进入正式候选。 |
| Transformer-lite | 深度时序候选 | `app/model/transformer_lite_sl60_compare.csv` | 已实验，当前 RankIC/Sharpe/回撤不满足替换门槛。 |
| ARIMA / TFT / N-HiTS / TSFM / Qlib / FinRL | 调研与适用性分析 | `app/docs/unimplemented_model_applicability.md` | 未强行纳入半成品实验，保留为后续扩展方向。 |

PPT 禁止写法：

- 不要写“已完整实现 ARIMA / TFT / N-HiTS / TSFM / Qlib / FinRL 实验”。
- 不要写“Transformer 已替换 LSTM 主线”。
- 不要把 aggressive 提交结果说成“新训练模型”，它是提交变体结果同步。

## 3. PPT 正式默认方案口径

PPT 中“最终方案/提交方案”建议拆成两层讲，避免混淆。

| 层级 | 应写内容 |
|---|---|
| 模型层 | 包内保留默认模型为 `LSTM sl20 + base_alpha_v3_rs_crowding_mini4 + risk_adjusted + pred`。 |
| 提交层 | 当前提交变体为 `aggressive_score_submission`，最终 `app/output/result.csv` 以 aggressive 满仓结果为准。 |
| 结果层 | 5 只股票为 `000792 / 600233 / 601669 / 600930 / 002463`，权重和 `1.000000`，可见 case-slice score `0.077484`。 |
| 复现层 | 本地 `app/output/result.csv` 与 Docker `test/output/result.csv` MD5 一致，见 `app/model/docker_consistency_check.md`。 |

PPT 可直接使用这句：

> 模型选择依据来自 LSTM sl20 的统一回测和模型对比；最终提交为了可见单切片得分采用 aggressive 变体同步结果。两者不是冲突关系：前者解释模型主线，后者解释提交文件。

## 4. PPT 图表引用路径

PPT 优先使用以下图表，路径和结论固定如下。

| PPT 页 | 图表/文件 | 路径 | 支撑结论 |
|---|---|---|---|
| 研究流程 | Demo 主流程图 | `app/docs/report_supplement_assets/demo_flowchart.png` | 项目不是脚本堆叠，而是研究、提交、Docker 三段闭环。 |
| 模型对比 | 正式模型对比图 | `app/docs/report_supplement_assets/formal_model_comparison_chart.png` | LSTM sl20 是当前综合最优主线。 |
| 模型对比表 | 正式模型对比表 | `app/model/formal_model_comparison/formal_model_comparison.csv` | 同协议比较 LSTM、LightGBM、XGBoost、Linear、Momentum。 |
| 市场阶段 | 市场阶段表现图 | `app/docs/report_supplement_assets/market_regime_analysis_chart.png` | 不同市场阶段表现不同，说明做了稳定性分析。 |
| 结果文件 | Demo 结果文件图 | `app/docs/report_supplement_assets/demo_result_files.png` | 关键产物路径清晰。 |
| Docker 复现 | Docker 一致性报告 | `app/model/docker_consistency_check.md` | 本地和容器输出一致。 |
| 未落地方向 | 适用性说明表 | `app/docs/unimplemented_model_applicability.md` | 开题中提到但未完整落地的方向已有正式解释。 |

PPT 图表页不要引用 `app/temp/` 下的临时文件；那是运行缓存，不作为展示证据。

## 5. Demo 固定展示顺序

Demo 和现场讲解按这个顺序走：

1. 打开 `app/docs/demo_flowchart.md`，讲研究链路、提交链路和 Docker 链路。
2. 打开 `app/docs/demo_key_commands.md`，讲 `run_research_pipeline.sh`、`run_submission.sh`、`freeze_submission.sh`、Docker 命令。
3. 打开 `app/docs/demo_result_files.md`，讲输入特征、最终 `result.csv`、模型对比表、提交冻结产物。
4. 打开 `app/model/formal_model_comparison/formal_model_comparison.md`，讲为什么选择 LSTM sl20。
5. 打开 `app/model/final_submission_snapshot.md`，讲当前提交变体、最终股票与权重。
6. 打开 `app/model/docker_consistency_check.md`，讲本地/Docker 输出一致。
7. 如被问到未落地模型，打开 `app/docs/unimplemented_model_applicability.md`。

如果启动 Streamlit Demo，只作为辅助展示，不替代以上证据链。

## 6. README / 报告 / PPT / Demo 对齐检查

| 检查项 | 统一结果 |
|---|---|
| 项目目标 | 沪深300收益预测与 Top-K 组合推荐。 |
| 正式提交文件 | `app/output/result.csv`。 |
| 当前提交变体 | `aggressive_score_submission`。 |
| 当前提交股票 | `000792 / 600233 / 601669 / 600930 / 002463`。 |
| 当前权重和 | `1.000000`。 |
| 包内默认模型 | `LSTM sl20 + base_alpha_v3_rs_crowding_mini4 + risk_adjusted + pred`。 |
| 未落地模型口径 | 调研与适用性分析，未纳入正式可复现实验链路。 |
| Docker 复现口径 | `app/output/result.csv` 与 `test/output/result.csv` 一致。 |
| 测试口径 | `pytest -q` 为 `26 passed`。 |

## 7. PPT 逐页建议

建议 PPT 控制在 10 到 12 页：

| 页码 | 标题 | 核心内容 |
|---:|---|---|
| 1 | 课题与目标 | 沪深300收益预测、Top-K 推荐、组合回测。 |
| 2 | 数据与特征 | 原始行情到统一特征表，强调时间顺序与防泄漏。 |
| 3 | 整体流程 | 使用 `demo_flowchart.png`。 |
| 4 | 模型清单 | 使用第 2 节模型清单。 |
| 5 | 模型选择依据 | 展示正式模型对比图/表。 |
| 6 | 回测与风险 | 展示收益、Sharpe、回撤、换手。 |
| 7 | 稳定性与市场阶段 | 展示市场阶段分析。 |
| 8 | 最终提交结果 | 展示 aggressive 变体、5 只股票和权重。 |
| 9 | Docker 与可复现 | 展示 Docker 一致性 PASS。 |
| 10 | 未落地方向说明 | 说明 ARIMA/TFT/N-HiTS/TSFM/Qlib/FinRL 的适用性判断。 |
| 11 | Demo 路线 | 按第 5 节顺序演示。 |
| 12 | 总结与后续 | 已完成闭环，后续补 TSFM/Qlib/FinRL 独立实验。 |
