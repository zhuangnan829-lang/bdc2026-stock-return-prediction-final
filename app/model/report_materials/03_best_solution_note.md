# 一页最优方案说明

## 当前最优方案名称

`base_technical_risk__risk_adjusted_sort__pred_weight__balanced_mt70`

## 方案定位

这是当前项目的正式冻结默认提交候选方案。  
它不是单纯追求某一次搜索里累计收益最高的版本，而是在收益率、稳定性、回撤和交易成本之间取得较好平衡的方案。

## 方案组成

### 1. 特征集

使用 `base_technical_risk` 特征集，共 `45` 个核心特征，包含：

- 基础收益率与动量特征
- 均线偏离与技术指标特征
- 波动率、风险调整动量与横截面 rank 特征

### 2. 训练目标

- 模型：`LightGBM`
- 训练目标：`cross_section_rank`

这样做的原因是比赛更关注“未来谁更值得选”，而不是单只股票点预测误差最小。

### 3. 选股逻辑

- Top-K：`5`
- 主候选池大小：`55`
- 风险过滤阈值：
  - `max_volatility_20d_pct = 0.82`
  - `max_volatility_5d_pct = 0.94`
  - `turnover_rate_lower_pct = 0.03`
  - `turnover_rate_upper_pct = 0.97`
  - `turnover_ratio_upper_pct = 0.95`

### 4. 排序与权重

- 排序策略：`risk_adjusted`
- 权重策略：`pred`
- 风险惩罚权重：`0.20`

即先根据预测得分筛选候选股票，再结合风险因子进行二次排序，最后使用预测收益归一化权重分配仓位。

### 5. 低换手执行

- `max_turnover = 0.70`
- `transaction_cost = 0.001`

这一设置的目标是在尽量保留收益率的同时，把过高换手带来的成本和回撤控制在更合理的水平。

## 为什么它是当前默认最优

根据目前所有实验结论：

- `base_technical_risk` 是最优特征集
- `risk_adjusted` 是最优排序策略
- 在最新固定候选池和风险阈值下，`pred` 比 `risk_adjusted` 权重更优
- `max_turnover=0.75` 虽然累计收益略高，但 `0.70` 在夏普和回撤上更平衡

因此，这套方案是当前最适合作为正式默认提交版的程序方案。

## 当前本地验证结论

当前这套默认方案对应的关键本地表现：

- 成本后累计收益：`0.125348`
- 成本后夏普：`1.578308`
- 成本后最大回撤：`-0.070621`
- 平均换手率：`0.700000`

## 程序落地位置

正式配置文件：

- `app/model/best_config.json`

提交候选配置：

- `app/model/default_submission_config.json`

配套程序：

- `app/code/src/train.py`
- `app/code/src/test.py`
- `app/code/src/backtest.py`
- `app/code/src/result_validator.py`
- `app/code/src/pre_submit_check.py`

## 当前建议

当前不建议再随意修改默认方案。  
如果后续还要继续优化，应以这套方案为基线做增量实验，并保持同一套回测和消融流程后再决定是否替换默认版。
