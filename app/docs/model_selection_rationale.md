# 模型选择与替换依据

本文档用于记录最终是否替换正式主线模型的判断依据。当前结论是：**继续保留 `LSTM sl20 + base_alpha_v3_rs_crowding_mini4 + risk_adjusted + pred` 作为正式提交主线**，`sl60`、`Transformer-lite`、`LightGBM`、`XGBoost`、`Momentum` 和 `rank blend` 只作为候选分支或对照基线。

## 1. 当前正式结论

- 正式主线：`LSTM sl20`
- 正式特征集：`base_alpha_v3_rs_crowding_mini4`
- 正式排序策略：`risk_adjusted`
- 正式权重策略：`pred`
- 当前状态：保留主线，不在最终交付前贸然切换到新模型。

保留原因：

1. `LSTM sl20` 已经完成训练、walk-forward、回测、正式推理、冻结配置和 Docker 彩排闭环。
2. 当前正式输出可以稳定生成 `app/output/result.csv`，且本机与 Docker 结果 MD5 一致。
3. 候选分支尚未在切片得分、RankIC、Top5、Sharpe、回撤、换手和 Docker 复现上同时不差。
4. 金融预测任务中，单一指标提升不足以证明可替换主线；必须同时满足收益、稳定性、风控和复现要求。

## 2. 替换主线的硬门槛

任何新模型或融合方案要替换 `LSTM sl20`，必须在同一数据切分、同一回测口径、同一提交生成链路下同时满足下面条件：

| 指标 | 替换要求 |
|---|---|
| 切片得分 | 不低于当前正式主线；若只提高切片得分但其他指标变差，不替换。 |
| RankIC | 平均 RankIC 不低于主线，且 worst fold 不更差。 |
| Top5 | Top5 平均收益、NDCG@5 或 HitRate@5 至少不劣于主线。 |
| Sharpe | 成本后 Sharpe 不低于主线。 |
| 回撤 | 成本后最大回撤不更差。 |
| 换手 | 平均换手不更高；如果换手更高，必须有显著收益补偿并通过成本压力测试。 |
| Docker 复现 | 必须能在 Docker 入口下生成一致的正式结果。 |
| 交付风险 | 必须同步更新 `default_submission_config.json`、`best_config.json`、`model_meta.json` 和冻结产物。 |

替换判断采用“一票否决”：只要新分支在稳定性、回撤、换手或复现上明显变差，就不替换正式主线。

## 3. 主要证据

### 3.1 当前 LSTM sl20 主线

来源：

- `app/model/final_submission_snapshot.md`
- `app/model/formal_model_comparison/formal_model_comparison.csv`
- `app/model/default_profile_backtest/backtest_summary.csv`
- `app/model/docker_consistency_check.md`
- `app/model/case_comparison/latest_score_compare.md`

关键指标：

| 指标 | 当前主线值 |
|---|---:|
| case slice score | 0.032984 |
| RankIC | 0.027982 |
| Top5 平均收益 | 0.007502 |
| 成本后累计收益 | 1.171246 |
| 成本后 Sharpe | 4.019488 |
| 成本后最大回撤 | -0.090067 |
| 平均换手 | 0.956671 |
| Docker 一致性 | PASS，local/docker result MD5 一致 |

当前主线超过参考案例当前输出得分，但尚未超过参考记录最好分数。因此，后续优化可以继续推进，但不应因为单一候选实验而推翻已有闭环。

### 3.2 sl40/sl60 序列长度分支

来源：`app/model/sequence_length_search/sl20_sl40_sl60_compare.csv`

| 实验 | RankIC | Top5 | Sharpe | 最大回撤 | 换手 | 累计收益 | 切片得分 | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `lstm_sl20` | 0.027982 | 0.007502 | 3.977455 | -0.087064 | 0.931137 | 1.085936 | 0.031491 | 主线 |
| `lstm_sl60` | 0.039856 | 0.010208 | 2.481363 | -0.099273 | 0.944684 | 0.596388 | 0.002946 | 候选，不替换 |
| `lstm_sl40` | -0.025075 | 0.000112 | 1.376108 | -0.095101 | 0.925798 | 0.244159 | 0.014807 | 不替换 |

结论：`sl60` 在 RankIC 和 Top5 上有提升，但 Sharpe、累计收益、最大回撤和切片得分明显不如 `sl20`，不能替换正式主线。`sl60` 可以保留为融合或稳健性研究分支。

### 3.3 Transformer-lite 分支

来源：`app/model/transformer_lite_sl60_compare.csv`

| 实验 | RankIC | Top5 | Sharpe | 最大回撤 | 换手 | 累计收益 | 切片得分 | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `lstm_sl20` | 0.027982 | 0.007502 | 3.977455 | -0.087064 | 0.931137 | 1.085936 | 0.031491 | 主线 |
| `transformer_lite_sl60` | -0.028891 | 0.004924 | 0.792688 | -0.121362 | 0.980408 | 0.111850 | -0.022891 | 候选对照，不替换 |

结论：`Transformer-lite` 当前 RankIC 为负，Sharpe 较低，回撤和换手更差，只能作为候选对照或 rank blend 输入，不能进入正式默认配置。

### 3.4 LightGBM/XGBoost/Linear/Momentum 基线

来源：

- `app/model/formal_model_comparison/formal_model_comparison.csv`
- `app/model/model_comparison/tree_model_comparison.csv`

| 模型 | RankIC | Top5 平均收益 | 回测累计收益 | Sharpe | 最大回撤 | 判断 |
|---|---:|---:|---:|---:|---:|---|
| `LSTM sl20` | 0.027982 | 0.007502 | 1.171246 | 4.019488 | -0.090067 | 主线 |
| `LightGBM` | -0.027417 | 0.002183 | 0.295839 | 1.711682 | -0.113166 | 对照，不替换 |
| `XGBoost` | -0.030145 | 0.000341 | 0.137837 | 1.045790 | -0.097146 | 对照，不替换 |
| `Linear Regression` | -0.028608 | 0.001038 | -0.041794 | -0.138544 | -0.186285 | 对照，不替换 |
| `Momentum` | -0.008896 | 0.001318 | 0.346502 | 1.199518 | -0.257034 | 规则基线，不替换 |

结论：树模型和规则基线可用于证明做过同协议对照，但当前不具备替换 `LSTM sl20` 的综合表现。

### 3.5 Rank blend 分支

来源：`app/model/rank_blend/blend_summary.csv`

当前 `rank_blend` 搜索中，`baseline__lstm_sl20` 是唯一被标记为 `adopted=True` 的方案；`app/model/rank_blend/adopted_blend_summary.csv` 为空，说明尚无融合方案通过采用门槛。

代表性结果：

| 方案 | RankIC | Sharpe | 最大回撤 | 换手 | 累计收益 | 采用状态 |
|---|---:|---:|---:|---:|---:|---|
| `baseline__lstm_sl20` | 0.027982 | 3.977455 | -0.087064 | 0.931137 | 1.085936 | adopted baseline |
| `lstm_sl20 0.6 + momentum 0.4` | 0.021553 | 2.479048 | -0.104854 | 0.938384 | 0.634444 | 不采用 |
| `lstm_sl20 0.7 + lightgbm 0.3` | 0.017071 | 2.798091 | -0.071888 | 0.990018 | 0.585250 | 不采用 |
| `lstm_sl20 0.6 + transformer_lite 0.4` | 0.008528 | 1.441126 | -0.116415 | 0.998333 | 0.229050 | 不采用 |

结论：融合方向值得继续保留，但当前没有方案同时改善收益、稳定性、集中度、换手和复现，因此不能替换正式主线。

## 4. 最终决策

当前最终交付决策如下：

| 分支 | 状态 | 原因 |
|---|---|---|
| `LSTM sl20` | 正式主线 | 证据闭环完整，综合指标最好，Docker 可复现。 |
| `LSTM sl60` | 候选分支 | RankIC/Top5 有改善，但切片得分、Sharpe、累计收益和回撤不满足替换门槛。 |
| `Transformer-lite sl60` | 候选对照 | 当前 RankIC、Sharpe、回撤和切片得分均不达标。 |
| `LightGBM/XGBoost` | 基线对照 | 同协议对照已完成，但综合表现弱于主线。 |
| `rank blend` | 候选方向 | 当前尚无融合方案通过采用门槛。 |

因此，最终提交仍使用 `LSTM sl20`。新分支只能进入实验排行榜、报告对照或后续研究，不直接改写正式默认配置。

## 5. 开题扩展方向的收敛说明

开题报告中还提到 `ARIMA / TFT / N-HiTS / TSFM / Qlib / FinRL`。这些方向当前不作为正式候选模型，不是因为忽略，而是因为没有在最终交付期内形成与主线同协议、同指标、可 Docker 复现的完整链路。

详细适用性判断已整理到：

- `app/docs/unimplemented_model_applicability.md`

最终口径如下：

| 方向 | 当前定位 | 不进入正式候选的原因 |
|---|---|---|
| ARIMA | 传统统计模型调研 | 单变量点预测定位明显，和横截面 Top-K 排序及组合回测目标不完全一致。 |
| TFT / N-HiTS | 深度时序后续方向 | 接入成本、调参周期和复现复杂度较高，当前未证明优于 LSTM 主线。 |
| TSFM | 前沿基础模型调研 | 外部模型/服务与离线复现约束较强，暂不适合作为正式提交依赖。 |
| Qlib | 量化框架参考 | 完整接入会重建数据层和回测协议，和当前自研闭环重复。 |
| FinRL | 强化学习框架参考 | 更适合连续交易环境，和当前单切片 Top-K 权重提交目标差异较大。 |

因此，结题报告应将这些方向表述为“已完成调研与适用性分析，保留为后续扩展”，而不是写成已经完成正式实验。

## 6. 后续可替换流程

如果后续出现新的强候选方案，应按以下流程替换：

1. 重新生成候选分支的 walk-forward 预测、fold 诊断、回测、切片得分和稳定性报告。
2. 写入 `app/model/experiment_leaderboard.csv`，并和 `LSTM sl20` 同口径比较。
3. 通过本文档第 2 节的全部硬门槛。
4. 更新 `app/model/default_submission_config.json`、`app/model/best_config.json`、`app/model/model_meta.json`。
5. 执行冻结流程并更新 `app/model/final_submission_snapshot.md`。
6. 执行 Docker 彩排，确认 `app/model/docker_consistency_check.md` 为 PASS。
7. 在本文档中补充替换记录和回退方案。

未完成以上流程前，任何候选模型都不得替换正式主线。
