# 3 分钟 Demo 主流程

## 目标

把当前项目从“很多脚本”讲成“一条完整主流程”：

1. 数据输入
2. 训练 / 推理
3. 回测
4. 冻结提交
5. aggressive 变体同步
6. Docker 入口

## 固定演示顺序

### Step 1. 数据输入

先说明项目输入不是黑盒，而是已经整理好的特征表：

- 训练特征：`app/temp/train_features.csv`
- 推理输出：`app/output/result.csv`

一句话讲法：

> 我们先把股票原始行情转成统一特征表，后面的训练、推理、回测和提交都共用这套输入口径。

### Step 2. 研究主流程

演示研究链路入口：

- `app/run_research_pipeline.sh`

一句话讲法：

> 研究阶段不是分散脚本，而是一条固定链路：训练、walk-forward 验证、冻结推理快照、本地回测、分折诊断。

### Step 3. 正式推理主流程

演示正式提交入口：

- `app/run_submission.sh`

一句话讲法：

> 正式入口默认不再现场训练，而是先使用冻结好的 LSTM 模型做推理，再按当前 aggressive 提交变体同步最终 `result.csv`，避免提交时出现训练漂移和结果口径漂移。

### Step 4. 冻结提交流程

演示冻结流程：

- `app/freeze_submission.sh`

一句话讲法：

> 当我们要准备最终提交版本时，会走冻结流程，把配置同步、结果校验和提交前自检串成一条闭环。

### Step 5. Docker 入口

演示容器入口：

- `docker build -t bdc2026 .`
- `docker compose up`

一句话讲法：

> 最后再把同一条正式推理链路放进 Docker 容器，保证本地演示路径和比赛提交路径一致。

## 3 分钟讲述模板

### 第 1 分钟：讲输入与研究链路

> 项目先把股票历史行情整理成统一特征表 `train_features.csv`。在研究阶段，我们只需要运行 `run_research_pipeline.sh`，它会自动完成训练、walk-forward 验证、冻结推理快照、本地回测和诊断分析。

### 第 2 分钟：讲正式推理与冻结提交

> 当模型方案确定后，我们不再把“训练”和“正式提交”混在一起。正式入口 `run_submission.sh` 默认使用冻结模型推理，并在检测到 `aggressive_score_submission` 后同步最终满仓结果。如果要准备最终提交包，就执行 `freeze_submission.sh`，它会自动做结果校验和提交前检查。

### 第 3 分钟：讲 Docker 与产物

> 最后，为了保证可复现，我们把同一条正式提交链路放进 Docker。老师最终看到的不是很多零散脚本，而是一条从输入特征、模型训练、策略回测到冻结提交、Docker 复现的完整主流程。

## 建议现场打开的文件

按顺序建议只打开这些：

1. `app/docs/demo_flowchart.md`
2. `app/docs/demo_key_commands.md`
3. `app/docs/demo_result_files.md`
4. `app/model/formal_model_comparison/formal_model_comparison.md`
5. `app/model/final_submission_snapshot.md`
6. `app/model/docker_consistency_check.md`
