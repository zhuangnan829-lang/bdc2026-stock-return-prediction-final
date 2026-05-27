# 股票分析预测项目小白说明

本文档是给第一次接触这个项目的人看的。目标不是讲很深的数学，而是把这个项目“用了什么语言、按什么思路做、每一步在干嘛、现在做到什么程度”讲清楚。

## 1. 这个项目是做什么的

这个项目的目标是：根据沪深 300 股票的历史行情数据，预测未来一段时间哪些股票更可能表现好，然后生成一个正式提交文件：

```text
app/output/result.csv
```

这个 `result.csv` 里面主要有两列：

| 列名 | 含义 |
|---|---|
| `stock_id` | 股票代码 |
| `weight` | 给这只股票分配多少仓位权重 |

简单理解：程序最后要回答的问题是：

> 在当前数据下，我应该选哪几只股票，每只股票买多少比例？

当前正式输出是最多选 5 只股票，并且总权重不能超过 1。

## 2. 项目主要用了什么语言和工具

### 2.1 主语言：Python

项目核心逻辑基本都是 Python 写的，主要文件在：

```text
app/code/src/
```

Python 在这里负责：

- 读取股票 CSV 数据
- 清洗字段和缺失值
- 计算技术指标、动量、波动率、换手率等特征
- 训练机器学习/深度学习模型
- 生成预测分数
- 选出最终股票
- 做回测、验证和结果检查

### 2.2 脚本语言：Shell / PowerShell / BAT

项目里有一些 `.sh`、`.ps1`、`.bat` 文件，它们不是模型本身，而是“启动按钮”。

| 文件 | 作用 |
|---|---|
| `app/train.sh` | 训练入口 |
| `app/test.sh` | 推理/预测入口 |
| `app/run_submission.sh` | 正式提交入口 |
| `app/freeze_submission.sh` | 冻结正式配置和提交快照 |
| `app/train.ps1`、`app/test.ps1` | Windows PowerShell 对应入口 |
| `app/run_train.bat`、`app/run_test.bat` | Windows 双击/命令行入口 |

你可以把这些脚本理解成：帮你自动按顺序运行多个 Python 文件。

### 2.3 主要 Python 库

依赖写在：

```text
requirements.txt
```

主要包括：

| 库 | 作用 |
|---|---|
| `pandas` | 处理表格数据，比如 CSV |
| `numpy` | 数值计算 |
| `scikit-learn` | 传统机器学习、指标计算、基线模型 |
| `lightgbm` | 树模型基线对比 |
| `torch` | 训练 LSTM、Transformer 等深度学习模型 |
| `matplotlib` | 画图，比如回测曲线 |
| `streamlit` | 做答辩展示 Demo |
| `pytest` | 自动化测试 |
| `joblib` | 保存/读取传统模型 |

### 2.4 Docker

Docker 用来模拟比赛或评测环境，确保项目不是只在你电脑上能跑，而是在容器里也能跑。

相关文件：

```text
Dockerfile
docker-compose.yml
```

简单理解：Docker 是一个“统一运行环境的盒子”，可以减少“我电脑能跑、别人电脑不能跑”的问题。

## 3. 整体思路是什么

项目整体思路可以拆成 7 步：

```text
原始行情数据
  -> 数据清洗
  -> 特征工程
  -> 模型训练
  -> walk-forward 验证
  -> 回测和风险分析
  -> 正式推理选股
  -> 生成 result.csv
```

换成更白话的说法：

1. 先拿到每只股票每天的开盘价、收盘价、成交量、换手率等数据。
2. 把中文字段统一改成程序好处理的英文字段。
3. 根据历史行情计算很多“特征”，比如最近 5 天涨跌、20 天波动率、成交量是否放大、是否过热。
4. 用历史数据告诉模型：以前出现这些特征时，后面股票表现怎么样。
5. 让模型学习规律。
6. 用没见过的时间段做验证，看模型是不是真的有用。
7. 最后用最新数据预测，选出排名靠前、风险可控的股票，写入 `result.csv`。

## 4. 数据从哪里来，怎么变成模型能用的东西

### 4.1 输入数据

主要数据在：

```text
app/data/
```

常见文件：

| 文件 | 作用 |
|---|---|
| `app/data/train.csv` | 训练数据 |
| `app/data/test.csv` | 预测/提交用数据 |
| `app/data/stock_data.csv` | 股票历史数据 |
| `app/data/hs300_stock_list.csv` | 沪深 300 股票列表 |

### 4.2 字段标准化

原始数据里可能是中文字段，比如：

| 原字段 | 程序内字段 |
|---|---|
| 股票代码 | `stock_id` |
| 日期 | `date` |
| 开盘 | `open` |
| 收盘 | `close` |
| 最高 | `high` |
| 最低 | `low` |
| 成交量 | `volume` |
| 成交额 | `amount` |
| 换手率 | `turnover_rate` |
| 涨跌幅 | `pct_change` |

这些转换主要在：

```text
app/code/src/featurework.py
```

### 4.3 特征工程

“特征”就是模型判断股票好坏时看的参考信息。

这个项目里计算了很多特征，例如：

| 特征类型 | 举例 | 白话解释 |
|---|---|---|
| 收益/动量 | `ret_1d`、`ret_5d`、`mom_10d` | 最近 1 天、5 天、10 天涨得怎么样 |
| 均线位置 | `close_to_ma_5d`、`close_to_ma_20d` | 当前价格和均线比是高还是低 |
| 波动率 | `volatility_5d`、`volatility_20d` | 最近股价波动大不大 |
| 成交量 | `volume_change_1d`、`volume_ratio_5d` | 成交量有没有突然放大 |
| 换手率 | `turnover_mean_5d`、`turnover_spike_5d` | 市场交易是否拥挤 |
| 横截面排名 | `rank_ret_5d`、`rank_volatility_20d` | 当天在所有股票里排第几 |
| 相对强弱 | `rel_hs300_mean_ret_5d` | 相比市场平均表现强不强 |
| 风险/拥挤 | `crowding_risk_5d`、`overheat_score` | 是否过热、是否风险偏高 |

训练时会生成：

```text
app/temp/train_features.csv
```

预测时会生成：

```text
app/temp/predict_features.csv
```

## 5. 模型是怎么做预测的

### 5.1 当前正式主线模型：LSTM

当前正式方案是：

```text
LSTM sl20 + base_alpha_v3_rs_crowding_mini4 + risk_adjusted + pred
```

拆开解释：

| 名称 | 含义 |
|---|---|
| `LSTM` | 一种适合处理时间序列的深度学习模型 |
| `sl20` | sequence length 20，模型会看过去 20 天的数据 |
| `base_alpha_v3_rs_crowding_mini4` | 当前正式使用的一组特征 |
| `risk_adjusted` | 排序时不仅看预测收益，也考虑风险 |
| `pred` | 权重主要根据模型预测分数来分配 |

为什么用 LSTM？

因为股票数据是按时间排列的。今天的走势和过去几天、十几天有关系。LSTM 的作用就是让模型看一段连续历史，而不是只看某一天。

训练入口：

```text
app/code/src/train_lstm.py
```

推理入口：

```text
app/code/src/test_lstm.py
```

### 5.2 训练目标

这个项目不是简单预测“明天涨几个点”，而是更关注“股票之间的相对排名”。

当前训练目标是：

```text
cross_section_rank
```

白话解释：

> 同一天有很多股票，模型重点学习哪几只股票未来更靠前，而不是死抠每只股票具体涨跌幅的小数点。

这个思路适合选股任务，因为最终提交也是选排名靠前的少数股票。

## 6. 训练流程是怎么样的

训练可以通过这个命令启动：

```bash
bash app/train.sh
```

实际过程大概是：

1. 检查训练需要的文件是否存在。
2. 执行初始化脚本 `app/init.sh`。
3. 运行 `featurework.py --mode train`，生成训练特征。
4. 运行 `train_lstm.py`，训练 LSTM 模型。
5. 做 walk-forward 验证。
6. 保存模型和指标。
7. 自动做一轮回测。

主要产物包括：

| 文件 | 作用 |
|---|---|
| `app/model/lstm_model.pt` | 训练好的 LSTM 模型 |
| `app/model/model_meta.json` | 模型使用了哪些特征、窗口长度、参数等信息 |
| `app/model/walk_forward_predictions.csv` | 滚动验证预测结果 |
| `app/model/walk_forward_metrics.csv` | 滚动验证指标 |
| `app/model/fold_diagnostics.csv` | 不同验证折的诊断结果 |

## 7. 什么是 walk-forward 验证

股票预测不能随机打乱训练集和测试集，因为时间顺序很重要。

错误做法是：

```text
拿未来数据训练，再预测过去
```

这样会造成数据泄漏。

本项目使用 walk-forward 验证，意思是：

```text
用更早的数据训练
预测后面一小段时间
再往后滚动
重复多次
```

白话理解：

> 模拟真实交易时“只能用过去预测未来”的情况。

这样得到的结果更可信。

## 8. 回测和风险分析在干嘛

模型预测出分数之后，还不能直接说它好。还要看：

- 如果真的按它选股，历史收益怎么样？
- 最大亏损回撤大不大？
- 换手率是不是太高？
- 交易成本扣掉后还赚钱吗？
- 单只股票权重是否过于集中？

回测入口：

```text
app/code/src/backtest.py
```

常见产物：

| 文件 | 作用 |
|---|---|
| `app/model/backtest_summary.csv` | 回测总体指标 |
| `app/model/backtest_daily_results.csv` | 每日收益明细 |
| `app/model/backtest_holdings.csv` | 每天持仓明细 |
| `app/model/backtest_report.md` | 回测报告 |
| `app/model/backtest_equity_*.png` | 净值曲线 |
| `app/model/backtest_drawdown_*.png` | 回撤曲线 |

当前主线指标记录在：

```text
app/model/final_submission_snapshot.md
```

其中包括：

| 指标 | 当前记录 |
|---|---:|
| 成本后累计收益 | `1.171246` |
| 成本后 Sharpe | `4.019488` |
| 成本后最大回撤 | `-0.090067` |
| 平均换手率 | `0.956671` |
| walk-forward RankIC | `0.027982` |
| walk-forward Top5 平均收益 | `0.007502` |

## 9. 正式预测和提交文件怎么生成

正式提交入口是：

```bash
bash app/run_submission.sh
```

它会做这些事：

1. 初始化环境。
2. 同步正式提交配置。
3. 检查配置是否一致。
4. 默认使用已经训练好的冻结模型，不重新训练。
5. 生成预测特征。
6. 加载 LSTM 模型。
7. 对最新日期股票打分。
8. 按风险调整后的排序选出候选股票。
9. 分配权重。
10. 生成 `app/output/result.csv`。
11. 校验 `result.csv` 格式。

输出文件：

| 文件 | 作用 |
|---|---|
| `app/output/result.csv` | 正式提交文件 |
| `app/output/predict_scores.csv` | 每只股票的预测分数 |
| `app/output/debug_candidates.csv` | 候选股票排序和筛选细节 |
| `app/output/slice_concentration_summary.csv` | 持仓集中度分析 |
| `app/output/slice_concentration_detail.csv` | 持仓集中度明细 |

## 10. 为什么要有配置文件

这个项目现在不是把参数随便写在某个脚本里，而是把正式参数集中放在配置文件中。

重要配置文件：

| 文件 | 作用 |
|---|---|
| `app/model/default_submission_config.json` | 正式推理默认配置，最重要 |
| `app/model/best_config.json` | 当前认为最好的方案配置 |
| `app/model/model_meta.json` | 模型元信息 |
| `app/model/final_submission_snapshot.md` | 冻结时的正式提交快照 |

这样做的好处是：

- 不容易忘记某个参数改过
- 文档、脚本、模型配置可以互相对齐
- 提交前可以自动检查配置是否一致
- 后面换模型或调参时有记录可追溯

配置一致性检查脚本：

```text
app/code/src/compare_config_consistency.py
```

## 11. Freeze 冻结流程是在干嘛

冻结命令：

```bash
bash app/freeze_submission.sh
```

它的作用是把当前正式方案固定下来，生成一份“交付快照”。

你可以理解为：

> 我现在决定用这个模型、这套参数、这个结果去提交，把它们打包留证据。

冻结产物：

| 文件/目录 | 作用 |
|---|---|
| `app/model/final_submission_snapshot.md` | 最终提交说明 |
| `app/model/submission_artifacts/` | 冻结后的提交产物目录 |
| `app/model/submission_artifacts/best_config.json` | 冻结时的最优配置 |
| `app/model/submission_artifacts/default_submission_config.json` | 冻结时的正式配置 |
| `app/model/submission_artifacts/model_meta.json` | 冻结时的模型信息 |

这个流程是为了防止最后提交前参数乱掉。

## 12. Docker 彩排是在干嘛

Docker 彩排命令：

```bash
docker build -t bdc2026 .
docker compose -p bdc2026_rehearsal up --abort-on-container-exit --force-recreate
docker compose -p bdc2026_rehearsal down
```

Docker 里会调用：

```text
app/data/run.sh
```

最终会生成：

```text
test/output/result.csv
```

它的意义是：

- 验证评测环境中也能跑
- 验证路径、依赖、入口文件没有问题
- 验证本机结果和 Docker 结果是否一致

本项目还写了对比脚本：

```text
app/code/src/compare_local_docker_result.py
```

用来比较本机的 `app/output/result.csv` 和 Docker 的 `test/output/result.csv`。

## 13. Streamlit Demo 是什么

Streamlit Demo 是一个展示页面，主要用于答辩或演示。

入口：

```text
app/demo/streamlit_app.py
```

启动方式：

```powershell
powershell -ExecutionPolicy Bypass -File .\app\demo\run_demo.ps1
```

或者：

```bash
bash app/demo/run_demo.sh
```

Demo 不是模型训练的核心，而是把项目结果、图表、指标、流程用更直观的方式展示出来。

## 14. 你之前到现在每一步都在干嘛

下面按项目推进顺序解释。

### 第 1 步：拿到比赛/案例原始材料

你最开始有原始压缩包和说明文件，例如：

```text
THU-BDC2026-82247deba1c7464295f66363efc94fd85549bfe0.zip
raw_get_stock_data_illustration.txt
raw_hs300_stock_list.csv
raw_stock_data.csv
tmp_notice.pdf
```

这一步是在理解任务要求：

- 要预测什么？
- 输入数据有哪些？
- 提交文件长什么样？
- 评测环境怎么运行？

### 第 2 步：整理项目目录

后来项目形成了比较清晰的目录：

```text
app/
  code/src/      核心代码
  data/          输入数据
  model/         模型、配置、实验结果
  output/        正式输出
  docs/          文档
  demo/          展示 Demo
tests/           自动化测试
scripts/         研究和搜索脚本
```

这一步是在把零散代码整理成一个正式项目。

### 第 3 步：准备运行环境

你维护了：

```text
requirements.txt
Dockerfile
docker-compose.yml
```

这一步是在解决：

- 需要安装哪些 Python 包？
- 别人拿到项目能不能复现？
- Docker 里能不能跑？

### 第 4 步：准备和切分数据

你把原始股票数据整理到：

```text
app/data/train.csv
app/data/test.csv
app/data/stock_data.csv
app/data/hs300_stock_list.csv
```

相关脚本包括：

```text
scripts/split_stock_data.py
app/data/run.sh
```

这一步是在把原始行情数据变成训练和预测能用的数据。

### 第 5 步：写特征工程

核心文件：

```text
app/code/src/featurework.py
```

这一步做的事情是：

- 读取 CSV
- 统一字段名
- 处理缺失值
- 计算收益、动量、波动率、成交量、换手、相对强弱等特征
- 训练模式生成 `train_features.csv`
- 预测模式生成 `predict_features.csv`

简单说：把普通行情表变成模型能读懂的“信号表”。

### 第 6 步：先做传统模型和基线

项目里保留了 LightGBM、XGBoost、Linear Regression、Momentum 等对比。

相关文件和目录：

```text
app/code/src/train.py
scripts/compare_tree_models.py
app/model/model_comparison/
app/model/xgboost_baseline/
app/model/baseline_lightgbm_same_protocol/
```

这一步是在回答：

> 如果不用 LSTM，普通机器学习或简单动量规则效果怎么样？

结果是：这些模型可以作为对照，但综合表现没有当前 LSTM 主线好。

### 第 7 步：训练 LSTM 主线模型

核心文件：

```text
app/code/src/train_lstm.py
app/code/src/lstm_utils.py
```

这一步做了：

- 把每只股票过去若干天组成一个序列
- 用 LSTM 学习时间序列规律
- 用 cross-section rank 目标学习股票排名
- 保存模型文件和模型信息

主要产物：

```text
app/model/lstm_model.pt
app/model/model_meta.json
```

### 第 8 步：做 walk-forward 验证

这一步也是训练流程的一部分。

它会生成：

```text
app/model/walk_forward_predictions.csv
app/model/walk_forward_metrics.csv
app/model/fold_diagnostics.csv
app/model/fold_daily_diagnostics.csv
```

这一步是在验证：

- 模型是否真的能用过去预测未来
- 不同时间段表现是否稳定
- 有没有某个阶段特别差

### 第 9 步：做本地回测

核心文件：

```text
app/code/src/backtest.py
```

这一步是在模拟：

> 如果历史上真的按模型选股，会赚多少？会亏多少？波动大不大？

产物包括：

```text
app/model/backtest_summary.csv
app/model/backtest_report.md
app/model/backtest_daily_results.csv
app/model/backtest_holdings.csv
```

这一步让模型结果不只是“预测指标好看”，还可以从投资组合角度检查风险和收益。

### 第 10 步：增加风险控制

相关文件：

```text
app/code/src/analyze_position_concentration.py
app/code/src/search_weight_cap.py
app/code/src/search_weight_blend.py
app/code/src/turnover_stress_test.py
```

你做了这些事情：

- 检查单只股票权重是否过高
- 搜索最大单票权重限制
- 测试换手率限制
- 测试交易成本压力
- 分析持仓集中度

这一步是在避免模型只会“冲分”，但实际组合太激进。

### 第 11 步：搜索候选池、排序和权重参数

相关脚本：

```text
scripts/search_inference_grid.py
scripts/search_inference_grid_refined.py
scripts/search_sort_weight_turnover_joint.py
scripts/search_weighting_risk_joint.py
app/code/src/search_candidate_pool.py
```

这一步是在调：

- 从多少只候选股票里选？
- 选前几只？
- 是纯按预测分数排，还是风险调整后排序？
- 权重是等权，还是按预测分数分配？
- 是否限制换手和单票权重？

简单说：模型给出分数后，还要研究“怎么把分数变成最终持仓”。

### 第 12 步：尝试更多模型和增强方案

你试过或保留了：

```text
LSTM sl20
LSTM sl40
LSTM sl60
Transformer-lite
LightGBM
XGBoost
Linear Regression
Momentum
rank blend
```

相关文件：

```text
app/code/src/run_sequence_length_search.py
app/code/src/train_transformer_lite.py
app/code/src/rank_blend.py
app/docs/model_selection_rationale.md
```

这一步是在比较：

> 有没有比当前 LSTM sl20 更好的方案？

当前结论是：保留 `LSTM sl20` 作为正式主线，其他方案作为候选或对照。

### 第 13 步：做特征消融和解释性分析

相关文件：

```text
scripts/run_ablation_study.py
app/code/src/run_alpha_v3_ablation.py
app/code/src/analyze_feature_drift.py
app/code/src/feature_importance_report.py
app/code/src/diagnose_misranked_samples.py
```

这一步是在回答：

- 哪些特征有帮助？
- 哪些特征可能没用？
- 某些股票为什么排错？
- 特征在不同时间段是否漂移？

这能让项目不是黑箱，而是能解释。

### 第 14 步：做数据泄漏检查

相关文件：

```text
app/code/src/check_data_leakage.py
app/model/data_leakage_check_report.md
```

这一步是在确认：

- 特征日期没有超过预测日期
- 标签是未来收益，不会被提前泄露
- 模型没有把答案列当成输入特征
- walk-forward 顺序符合“过去预测未来”

这对股票预测特别重要，因为一旦泄漏未来信息，模型表现会虚高。

### 第 15 步：建立正式提交链路

相关入口：

```text
app/run_submission.sh
app/test.sh
app/code/src/test_lstm.py
```

这一步是在让项目可以稳定生成：

```text
app/output/result.csv
```

同时还生成：

```text
app/output/predict_scores.csv
app/output/debug_candidates.csv
```

这说明项目不只是研究，还能产出正式提交文件。

### 第 16 步：增加结果校验

相关文件：

```text
app/code/src/result_validator.py
app/code/src/pre_submit_check.py
app/code/src/check_required_files.py
```

这一步是在检查：

- `result.csv` 列名对不对
- 股票代码是不是 6 位
- 最多是不是 5 只股票
- 权重是否非负
- 权重和是否不超过 1
- 关键文件是否都存在
- 配置是否一致

这是为了减少最后提交时低级错误。

### 第 17 步：冻结正式配置

相关文件：

```text
app/freeze_submission.sh
app/model/final_submission_snapshot.md
app/model/submission_artifacts/
```

这一步是在把当前正式方案固定下来，方便提交和答辩时说明：

- 用的是哪个模型
- 用的是哪套参数
- 输出结果是什么
- 回测指标是多少
- 配置来源在哪里

### 第 18 步：做 Docker 离线彩排

相关文件：

```text
Dockerfile
docker-compose.yml
app/docker_rehearsal.sh
app/code/src/compare_local_docker_result.py
app/model/docker_consistency_check.md
```

这一步是在验证：

- Docker 能不能构建成功
- 容器里能不能跑完整流程
- 容器生成的结果和本机是否一致

这对正式交付非常关键。

### 第 19 步：建立自动化测试

测试目录：

```text
tests/
```

覆盖内容包括：

- 结果校验
- 回测基础逻辑
- 特征工程 smoke test
- 配置文件检查
- README 路径检查
- Docker 对比逻辑
- 统一 CLI 入口
- 实验工具
- 特征漂移报告

这一步是在给项目加“安全网”，以后改代码不容易把已有功能弄坏。

### 第 20 步：增加 CI

相关文件：

```text
.github/workflows/basic-ci.yml
```

CI 的作用是自动安装依赖、运行测试、检查关键文件。

简单说：每次代码变化后，系统可以自动帮你检查有没有明显问题。

### 第 21 步：整理文档和报告材料

文档目录：

```text
app/docs/
```

重要文档包括：

| 文件 | 作用 |
|---|---|
| `experiment_result_index.md` | 实验结果索引 |
| `reproducibility_guide.md` | 复现说明 |
| `environment_snapshot.md` | 环境快照 |
| `final_delivery_checklist.md` | 最终交付清单 |
| `model_selection_rationale.md` | 为什么选当前模型 |
| `program_completion_review.md` | 程序完善复盘 |
| `demo_3min_main_flow.md` | 3 分钟演示主线 |

这一步是在把项目从“代码能跑”升级成“别人能看懂、能复现、能答辩”。

### 第 22 步：做 Streamlit 展示 Demo

相关文件：

```text
app/demo/streamlit_app.py
app/demo/run_demo.ps1
app/demo/run_demo.sh
```

这一步是在方便现场展示：

- 当前模型是什么
- 最终选股结果是什么
- 回测图表怎么看
- 实验对比结论是什么
- 为什么没有换成其他模型

## 15. 当前项目做到什么程度

当前项目已经具备：

- 可运行的训练入口
- 可运行的正式预测入口
- 可生成正式提交文件 `result.csv`
- LSTM 主线模型
- 多个候选模型对比
- walk-forward 验证
- 本地回测
- 风险控制分析
- 配置一致性检查
- 提交结果校验
- Docker 彩排
- 自动化测试
- 答辩展示 Demo
- 项目说明和交付文档

所以它已经不是一个单独脚本，而是一个比较完整的股票收益预测工程。

## 16. 最重要的几个入口

如果你只记几个命令，记下面这些：

### 安装依赖

```bash
pip install -r requirements.txt
```

### 完整研究复现

```bash
bash app/run_research_pipeline.sh
```

### 正式生成提交结果

```bash
bash app/run_submission.sh
```

### 校验提交结果

```bash
python app/code/src/result_validator.py --result_path app/output/result.csv
```

### 冻结正式提交快照

```bash
bash app/freeze_submission.sh
```

### 启动展示 Demo

```powershell
powershell -ExecutionPolicy Bypass -File .\app\demo\run_demo.ps1
```

## 17. 一句话总结

这个项目用 Python 做股票数据处理、特征工程和 LSTM 模型预测，用 Shell/PowerShell 做流程入口，用 Docker 保证交付环境可复现，用 pytest 和校验脚本保证结果不容易出错。整体流程是：先把原始行情数据变成特征，再训练模型学习股票排名，再通过回测和风险分析筛选方案，最后生成符合提交要求的 `result.csv`。

