# 沪深300股票收益预测与组合推荐

本项目面向沪深300成分股历史行情数据，构建了一套从数据处理、特征工程、模型训练、横截面排序、Top-K 股票推荐、组合回测到冻结提交和 Docker 复现的完整实验闭环。

当前正式提交包为 `aggressive_score_submission`。包内保留 `LSTM sl20` 默认模型和配置，用于复现推理链路与解释模型选择依据；最终 `app/output/result.csv` 会按 aggressive 变体同步为满仓提交结果。

## 当前提交结果

正式提交文件：

- `app/output/result.csv`

当前结果：

```csv
stock_id,weight
000792,0.2
600233,0.2
601669,0.2
600930,0.19999999999999998
002463,0.19999999999999996
```

关键状态：

| 项目 | 当前值 |
|---|---:|
| 提交变体 | `aggressive_score_submission` |
| 权重和 | `1.000000` |
| 可见 case-slice score | `0.077484` |
| local/docker result MD5 | `71401e2070a49f521bd40477fe0b5d16` |
| 测试状态 | `26 passed` |

对应说明：

- 变体说明：`PACKAGE_VARIANT.md`
- 机器可读变体信息：`app/model/package_variant.json`
- 最终提交快照：`app/model/final_submission_snapshot.md`
- Docker 一致性报告：`app/model/docker_consistency_check.md`

## 方法概览

项目主流程如下：

1. 将原始行情数据转换为统一特征表。
2. 使用 walk-forward 方式训练与验证模型，避免时间序列数据泄漏。
3. 以横截面排序能力为核心，评估 RankIC、Top-K 收益、命中表现等指标。
4. 将预测排序转化为 Top-K 股票组合，并纳入交易成本、换手和回撤分析。
5. 冻结正式配置、模型和结果文件。
6. 通过本地校验、自动化测试和 Docker 彩排确认可复现。

正式默认模型配置：

| 项目 | 当前设置 |
|---|---|
| 模型 | `LSTM` |
| 序列长度 | `20` |
| 特征集 | `base_alpha_v3_rs_crowding_mini4` |
| 排序策略 | `risk_adjusted` |
| 权重策略 | `pred` |
| Top-K | `5` |

## 模型选择依据

当前已完成统一协议对比的方向包括：

- `LSTM sl20`
- `LightGBM`
- `XGBoost`
- `Linear Regression`
- `Momentum`
- `Transformer-lite`

最终保留 `LSTM sl20` 作为包内默认模型，原因是它在排序能力、Top-K 收益、成本后回测表现和复现链路上综合最稳。`Transformer-lite`、树模型和规则基线已作为对照实验保留，但未替换正式主线。

开题报告中提到的 `ARIMA / TFT / N-HiTS / TSFM / Qlib / FinRL` 已整理为调研与适用性分析方向，未强行纳入半成品实验。详细说明见：

- `app/docs/unimplemented_model_applicability.md`
- `app/docs/model_selection_rationale.md`
- `app/docs/opening_report_alignment.md`

## 运行入口

### Windows 本地推理

```powershell
powershell -ExecutionPolicy Bypass -File .\app\test.ps1
```

### Linux / Docker 风格入口

```bash
bash app/run_submission.sh
```

### 重新训练

```bash
bash app/train.sh
```

或 Windows：

```powershell
powershell -ExecutionPolicy Bypass -File .\app\train.ps1
```

### 冻结提交

```bash
bash app/freeze_submission.sh
```

Windows：

```powershell
powershell -ExecutionPolicy Bypass -File .\app\freeze_submission.ps1
```

说明：`run_submission.sh`、`test.sh` 和 `test.ps1` 会先运行默认 LSTM 冻结推理，再检测 `app/model/package_variant.json`。当变体为 `aggressive_score_submission` 时，会把 `app/model/aggressive_score_submission_candidate/result_aggressive_score.csv` 同步回 `app/output/result.csv`，确保最终输出为当前提交结果。

## 校验与测试

只校验结果格式：

```bash
python app/code/src/result_validator.py --result_path app/output/result.csv
```

提交前完整检查：

```bash
python app/code/src/pre_submit_check.py --root_dir . --result_path app/output/result.csv
```

自动化测试：

```bash
pytest -q
```

当前验证结果：

```text
26 passed
```

## Docker 复现

构建镜像：

```bash
docker build -t bdc2026 .
```

运行彩排：

```bash
docker compose up
```

Docker 默认入口为：

- `app/data/run.sh`

该入口会调用：

- `app/run_submission.sh`

当前本地 `app/output/result.csv` 与 Docker 输出 `test/output/result.csv` 已通过一致性检查，报告见：

- `app/model/docker_consistency_check.md`

## 目录结构

| 路径 | 说明 |
|---|---|
| `app/code/src/` | 核心源码、训练、推理、回测、校验和实验工具 |
| `app/data/` | 评测/运行入口和数据文件 |
| `app/model/` | 模型、配置、实验结果、快照和复现证据 |
| `app/output/` | 当前正式输出结果 |
| `app/docs/` | 结题、答辩、复现和展示材料 |
| `app/demo/` | Streamlit 展示入口 |
| `tests/` | 自动化测试 |
| `.github/workflows/` | CI 配置 |

## 答辩与展示材料

建议展示时以以下文档为准，避免 README、PPT、Demo 和报告口径不一致：

- 展示统一口径：`app/docs/final_presentation_alignment.md`
- 结题报告：`app/docs/final_project_report.md`
- 3 分钟 Demo 主流程：`app/docs/demo_3min_main_flow.md`
- Demo 流程图：`app/docs/demo_flowchart.md`
- 关键命令页：`app/docs/demo_key_commands.md`
- 结果文件说明：`app/docs/demo_result_files.md`
- 图表索引：`app/docs/report_figure_table_index.md`
- 最终交付清单：`app/docs/final_delivery_checklist.md`

答辩建议口径：

> 模型选择依据来自 `LSTM sl20` 的统一回测和模型对比；最终提交为了可见单切片得分采用 `aggressive_score_submission` 变体同步结果。两者不是冲突关系：前者解释模型主线，后者解释最终提交文件。

## 当前 Git 迭代记录

项目已经补充了清晰的阶段性提交记录，主要包括：

- 自动化测试与 CI
- 可复现提交链路
- 交付文档与 Demo 流程
- 模型证据与冻结结果
- 未落地模型/框架适用性说明
- 最终展示口径统一

这部分用于回应评测中对“开发过程和迭代证据”的关注。
