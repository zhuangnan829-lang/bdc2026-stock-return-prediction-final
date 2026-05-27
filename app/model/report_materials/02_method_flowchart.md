# 一页方法流程图

## 方法总流程

```mermaid
flowchart TD
    A[原始数据获取\nstock_data.csv / hs300_stock_list.csv] --> B[数据预处理\n字段统一 日期解析 排序 缺失值处理]
    B --> C[时间切分\ntrain.csv / test.csv]
    C --> D[标签构造\nfuture_open_1 future_open_5 target_return]
    D --> E[特征工程\nbase / technical / risk]
    E --> F[模型训练\nLightGBM\ncross_section_rank]
    F --> G[Walk-forward验证\nRankIC Top5收益]
    G --> H[推理打分\npred_return]
    H --> I[候选池筛选\n主候选池 + 风险过滤]
    I --> J[排序策略\nrisk_adjusted sort]
    J --> K[权重分配\npred weight]
    K --> L[低换手执行\nmax_turnover=0.70]
    L --> M[输出result.csv]
    L --> N[本地回测\n收益 回撤 夏普 换手 交易成本]
    N --> O[消融实验\n特征 排序 权重 换手]
    O --> P[冻结默认提交方案]
```

## 模块解释

### 1. 数据层

- 输入沪深300成分股历史行情数据和成分股清单
- 统一股票代码、日期和数值字段
- 按时间顺序构建训练集和本地验证集

### 2. 标签层

- 以未来开盘价构造监督学习目标
- 标签严格对应：
  - `T+1` 开盘买入
  - `T+5` 开盘卖出

### 3. 特征层

- 基础收益率与动量特征
- 技术指标与均线偏离特征
- 风险波动与横截面 rank 特征

### 4. 模型层

- 使用 `LightGBM` 作为主模型
- 训练目标为 `cross_section_rank`
- 用 `walk-forward` 验证稳定性

### 5. 策略层

- 先按预测值形成候选池
- 再做风险过滤
- 用风险调整排序选择股票
- 用风险调整权重分配仓位
- 用低换手执行压缩交易成本

### 6. 评估层

- 本地回测输出累计收益、回撤、夏普、换手率和成本后收益
- 通过消融实验验证：
  - 哪组特征最好
  - 哪种排序最好
  - 哪种权重最好
  - 哪个换手约束最好

## 当前冻结主线

当前项目主线已经固定为：

`数据预处理 -> 完整特征集 -> LightGBM排序训练 -> 风险调整排序 -> 预测值权重分配 -> 低换手执行 -> result.csv`
