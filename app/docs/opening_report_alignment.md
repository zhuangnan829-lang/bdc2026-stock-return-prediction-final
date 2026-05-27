# 开题承诺对齐表

本文档用于对齐开题/中期材料中提到的模型、展示形式与当前仓库的真实落地状态，方便答辩时快速回答“写了什么、做了什么、为什么这样收敛”。

## 1. 对齐结论

- 当前**正式默认方案**已经冻结为 `LSTM sl20 + base_alpha_v3_rs_crowding_mini4 + risk_adjusted + pred`。
- `LSTM / LightGBM / XGBoost / 动量基线 / Streamlit Demo` 已落地，且有对应结果文件或展示入口。
- `Transformer` 已有基线实验与结果文件，但**未进入正式候选**。
- `ARIMA / TFT / N-HiTS / TSFM / Qlib / FinRL` 当前**未形成统一可复现实验链路**，主要属于调研、适用性分析或后续补充方向。

## 2. 模型与展示形式对齐表

| 项目 | 开题/中期材料中的角色 | 当前状态 | 是否纳入正式候选 | 对应结果文件/入口 | 说明 |
|---|---|---|---|---|---|
| LSTM | 主线深度学习模型 | 已完成 | 是 | `app/model/lstm_model.pt`、`app/model/model_meta.json`、`app/model/walk_forward_predictions.csv`、`app/model/backtest_summary.csv` | 当前正式默认方案使用 `LSTM sl20`。 |
| LightGBM | 机器学习基线 | 已完成 | 否 | `app/model/baseline_lightgbm_same_protocol/`、`app/model/formal_model_comparison/formal_model_comparison.csv` | 已纳入统一模型对比表。 |
| XGBoost | 树模型基线 | 已完成 | 否 | `app/model/xgboost_baseline/`、`app/model/formal_model_comparison/formal_model_comparison.csv` | 作为同特征集树模型对照，已切换到正式方案特征集。 |
| 动量基线 | 规则基线 | 已完成 | 否 | `app/model/formal_model_comparison/formal_model_comparison.csv` | 以 `Momentum (mom_5d)` 形式进入正式对比表。 |
| Linear Regression | 线性基线 | 已完成 | 否 | `app/model/baseline_linear_same_protocol/`、`app/model/formal_model_comparison/formal_model_comparison.csv` | 作为同特征集线性参考。 |
| Transformer | 深度时序备选模型 | 已有实验 | 否 | `app/model/transformer_baseline/`、`app/model/model_comparison/model_comparison_summary.csv` | 有训练与回测产物，但未收敛为正式方案。 |
| Streamlit Demo | 答辩/展示入口 | 已完成 | 不适用 | `app/demo/streamlit_app.py`、`app/demo/run_demo.ps1` | 当前用于展示正式方案、模型对比、回测与诊断。 |
| ARIMA | 传统统计模型 | 未落地 | 否 | 无统一实验产物 | 当前仓库中未见独立 ARIMA 训练/回测链路。 |
| TFT | 深度时序模型 | 未落地 | 否 | 无统一实验产物 | 当前以调研和方向分析为主。 |
| N-HiTS | 深度时序模型 | 未落地 | 否 | 无统一实验产物 | 当前以调研和方向分析为主。 |
| TSFM | 前沿基础模型方向 | 调研为主 | 否 | `app/docs/midterm_progress_report.md` 中有方向说明 | 当前缺少统一实验表，建议保留为“方向分析”。 |
| Qlib | 参考框架 | 未落地 | 否 | 无统一实验产物 | 目前未形成基于 Qlib 的可复现实验链路。 |
| FinRL | 参考框架 | 未落地 | 否 | 无统一实验产物 | 目前未形成基于 FinRL 的可复现实验链路。 |

## 3. 未完整落地方向的正式说明

开题报告中提到的 `ARIMA / TFT / N-HiTS / TSFM / Qlib / FinRL` 已统一收敛为“调研与适用性分析”口径，不作为当前正式提交链路的一部分。详细说明见：

- `app/docs/unimplemented_model_applicability.md`

核心原因如下：

| 方向 | 未纳入正式链路的核心原因 |
|---|---|
| ARIMA | 更适合单变量点预测，难以自然服务于沪深300横截面 Top-K 排序和组合推荐。 |
| TFT / N-HiTS | 需要额外时序框架、数据集组织和较长调参周期，短期内不如已闭环的 LSTM 主线稳定。 |
| TSFM | 外部模型/服务、依赖和离线复现约束较强，与当前比赛提交环境不完全匹配。 |
| Qlib | 完整接入会重建数据层、日历和回测协议，和当前自研闭环重复，交付期内风险较高。 |
| FinRL | 更适合连续交易强化学习环境，和当前单切片 Top-K 权重提交目标不完全一致。 |

因此，结题报告和 PPT 中应明确写为：这些方向已完成调研和适用性判断，未强行补半成品实验；正式实验证据以已经完成统一协议的 LSTM、LightGBM、XGBoost、Linear、Momentum 和 Transformer-lite 为准。

## 4. 为什么最后正式方案是 LSTM sl20

### 4.1 直接原因

根据当前正式模型对比结果：

- `app/model/formal_model_comparison/formal_model_comparison.csv`
- `app/model/formal_model_comparison/formal_model_comparison.md`
- `app/model/final_submission_snapshot.md`

`LSTM sl20` 在当前统一口径下同时具备：

1. 更好的 `RankIC`
2. 更高的 `Top5平均收益`
3. 更高的回测累计收益
4. 更高的 `Sharpe`

这意味着它不仅排序能力更强，而且更能把排序优势转化为组合收益。

### 4.2 相比历史主线 `sl10` 的变化

当前冻结默认方案已从历史参考主线 `LSTM sl10` 切换到 `LSTM sl20`。原因主要有两点：

1. 在相同 walk-forward 协议下，`sl20` 的整体排序表现更稳。
2. `sl20` 对第一折阶段性退化问题有一定缓解，因此更适合作为正式默认方案。

### 4.3 为什么其他模型没有进入正式候选

| 模型 | 未进入正式候选的主要原因 |
|---|---|
| LightGBM | 同特征集下整体排序能力与收益转化弱于当前 LSTM 主线。 |
| XGBoost | 即使切换到正式方案同特征集，整体收益与 RankIC 仍不占优。 |
| 动量基线 | 规则简单，NDCG/命中率不差，但收益转化与稳定性不足。 |
| Linear Regression | 作为线性参考已完成使命，但整体表现明显弱于 LSTM。 |
| Transformer | 已有实验，但当前结果未显示出超过 LSTM 主线的综合优势。 |

## 5. 答辩时建议这样表述

可以直接用下面这段逻辑：

> 开题阶段我们把模型方向铺得比较宽，包含传统统计模型、机器学习模型、深度时序模型和参考框架。  
> 当前仓库中真正形成统一训练、验证、回测和结果沉淀链路的，是 LSTM 主线及其基线对照。  
> 其中 `LSTM sl20` 在排序能力、推荐收益和成本后回测表现上综合最优，因此被冻结为正式默认方案。  
> 其他方向里，`Transformer` 已做过基线实验，但暂未超过当前主线；`ARIMA / TFT / N-HiTS / TSFM / Qlib / FinRL` 当前以调研和适用性分析为主，未纳入正式提交链路。  

## 6. 后续建议

- 若时间充足，可在 PPT 中加入一页“未纳入正式实验链路的模型/框架说明”，直接引用 `app/docs/unimplemented_model_applicability.md` 的表格。
- README、PPT、结题报告中的模型清单，建议以本对齐表为准统一口径。
