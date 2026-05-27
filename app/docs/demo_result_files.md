# Demo 结果文件说明

## 1. 输入文件

### `app/temp/train_features.csv`

作用：

- 训练与验证统一使用的特征表
- 体现数据输入已经标准化

现场讲法：

> 项目不是直接拿原始行情硬喂模型，而是先统一加工成结构化特征表。

## 2. 正式预测结果

### `app/output/result.csv`

作用：

- 最终正式提交文件
- 记录最终推荐股票及权重

现场讲法：

> 这是最终提交给比赛平台的核心文件。当前包会先运行默认 LSTM 冻结推理，再按 `aggressive_score_submission` 变体同步为最终提交结果。

## 3. 模型对比结果

### `app/model/formal_model_comparison/formal_model_comparison.csv`

作用：

- 展示 LSTM、LightGBM、Linear Regression、Momentum、XGBoost 的统一对比结果
- 说明主线模型不是只和自己比

现场讲法：

> 我们已经把多个基线和正式主线放进同一张表，能清楚说明改进相对哪些基线成立。

## 4. 市场阶段分析结果

### `app/model/market_regime_analysis/fold_stage_performance.csv`
### `app/model/market_regime_analysis/market_regime_report.md`

作用：

- 解释为什么不同 fold 表现差异明显
- 说明高波动 / 低波动、趋势 / 震荡下模型表现不同

现场讲法：

> 现在我们不只是在描述“稳定性波动”，而是能指出波动主要集中在哪类市场阶段。

## 5. 提交冻结产物

### `app/model/submission_artifacts/`

作用：

- 保存正式冻结模型、配置快照和提交侧最小必需产物

现场讲法：

> 这部分让正式提交不再依赖现场训练，保证结果可复现、可审计。

## 6. 老师最值得看的三个结果

如果时间只有几十秒，建议只展示：

1. `app/output/result.csv`
2. `app/model/formal_model_comparison/formal_model_comparison.md`
3. `app/model/final_submission_snapshot.md`
