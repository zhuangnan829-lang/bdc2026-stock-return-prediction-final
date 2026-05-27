# 赛题与代码规范合规性检查

## 结论

截至当前仓库状态，**程序已经基本满足“结果文件格式、训练/推理脚本、Docker 基本打包、提交配置一致性”这几类核心要求**，但**还不能稳妥地说已经 100% 符合所有代码要求**。

更准确地说：

- **当前可确认已满足的部分**：训练脚本、推理脚本、`result.csv` 格式约束、配置冻结、一致性自检、基础 Docker 打包链路。
- **当前仍存在的主要风险**：与赛事测试容器/参考测试方式的兼容层还不够完整，提交镜像内容偏重，Docker 默认入口更偏“研发彩排”而不是“最小提交入口”，还有少量检查项还不够严格。

因此，目前状态更适合定义为：

> **“已经接近正式提交可用，但还需要再做一轮面向赛事评测环境的收口。”**

## 本次检查依据

本次判断综合了以下材料：

- 你提供的赛题与代码规范文件：
  - `d:\Desktop\2026大数据挑战赛-赛题描述.pdf`
  - `d:\Desktop\2026大数据挑战赛-代码规范.pdf`
- 参考压缩包中的测试与运行材料：
  - `_case_zip/THU-BDC2026-82247deba1c7464295f66363efc94fd85549bfe0/test/赛事方测试方法.md`
  - `_case_zip/THU-BDC2026-82247deba1c7464295f66363efc94fd85549bfe0/test/test.py`
  - `_case_zip/THU-BDC2026-82247deba1c7464295f66363efc94fd85549bfe0/test/test_windows.py`
  - `_case_zip/THU-BDC2026-82247deba1c7464295f66363efc94fd85549bfe0/docker-compose.yml`
  - `_case_zip/THU-BDC2026-82247deba1c7464295f66363efc94fd85549bfe0/GUIDE.md`
- 当前仓库的正式提交路径与校验脚本：
  - [app/readme.md](D:/Desktop/股票分析预测代码/app/readme.md:1)
  - [app/test.sh](D:/Desktop/股票分析预测代码/app/test.sh:1)
  - [app/test.ps1](D:/Desktop/股票分析预测代码/app/test.ps1:1)
  - [app/freeze_submission.sh](D:/Desktop/股票分析预测代码/app/freeze_submission.sh:1)
  - [app/freeze_submission.ps1](D:/Desktop/股票分析预测代码/app/freeze_submission.ps1:1)
  - [Dockerfile](D:/Desktop/股票分析预测代码/Dockerfile:1)
  - [requirements.txt](D:/Desktop/股票分析预测代码/requirements.txt:1)
  - [app/code/src/result_validator.py](D:/Desktop/股票分析预测代码/app/code/src/result_validator.py:1)
  - [app/code/src/pre_submit_check.py](D:/Desktop/股票分析预测代码/app/code/src/pre_submit_check.py:1)

补充说明：

- 当前环境下无法直接自动抽取两份 PDF 的全文，因此这里的判断更多是基于你提供的文档主题、参考压缩包测试方式、以及仓库现状做出的**工程化合规检查**。
- 如果后续你希望做“逐条对照 PDF 原文”的最终版核查，建议再补一轮人工逐条复核。

## 当前已经满足或基本满足的要求

### 1. 训练、推理、初始化脚本已经具备

当前仓库已经有完整脚本链：

- [app/init.sh](D:/Desktop/股票分析预测代码/app/init.sh:1)
- [app/train.sh](D:/Desktop/股票分析预测代码/app/train.sh:1)
- [app/test.sh](D:/Desktop/股票分析预测代码/app/test.sh:1)
- [app/init.ps1](D:/Desktop/股票分析预测代码/app/init.ps1:1)
- [app/train.ps1](D:/Desktop/股票分析预测代码/app/train.ps1:1)
- [app/test.ps1](D:/Desktop/股票分析预测代码/app/test.ps1:1)

这说明你已经满足了“项目可以按固定入口初始化、训练、预测”的基本代码组织要求。

### 2. `result.csv` 输出格式已经被显式约束

[app/code/src/result_validator.py](D:/Desktop/股票分析预测代码/app/code/src/result_validator.py:1) 已明确检查：

- 列名必须为 `stock_id,weight`
- 最多 5 行
- `stock_id` 必须为 6 位字符串
- `weight` 必须非负
- 权重和必须不超过 `1.0`

这与竞赛类提交文件的常见硬约束是对齐的，属于当前仓库的一个明确优点。

### 3. 提交默认配置已经基本冻结

以下文件已经形成统一口径：

- [app/model/best_config.json](D:/Desktop/股票分析预测代码/app/model/best_config.json:1)
- [app/model/default_submission_config.json](D:/Desktop/股票分析预测代码/app/model/default_submission_config.json:1)
- [app/model/model_meta.json](D:/Desktop/股票分析预测代码/app/model/model_meta.json:1)
- [app/model/final_submission_snapshot.md](D:/Desktop/股票分析预测代码/app/model/final_submission_snapshot.md:1)

而且当前正式默认已经统一到 `rp-60` 这套配置，这一点对“代码规范中的可复现性、一致性”是加分项。

### 4. 已经具备提交前自检机制

[app/code/src/pre_submit_check.py](D:/Desktop/股票分析预测代码/app/code/src/pre_submit_check.py:1) 已经能检查：

- 必需文件是否存在
- `result.csv` 是否符合格式
- 配置快照之间是否一致

这说明你不是只“能跑出结果”，而是已经开始把提交过程工程化了。

### 5. 已经具备 Docker 基础打包能力

[Dockerfile](D:/Desktop/股票分析预测代码/Dockerfile:1) 和 [requirements.txt](D:/Desktop/股票分析预测代码/requirements.txt:1) 已存在，且容器内默认工作目录、依赖安装、脚本执行入口都已定义。

这说明你已经满足“可构建离线镜像”的基础门槛。

### 6. 已经具备一键冻结提交流程

你现在已经有：

- [app/freeze_submission.sh](D:/Desktop/股票分析预测代码/app/freeze_submission.sh:1)
- [app/freeze_submission.ps1](D:/Desktop/股票分析预测代码/app/freeze_submission.ps1:1)

它们会依次做：

- 同步正式配置
- 跑推理
- 校验 `result.csv`
- 执行提交前检查
- 更新对比报告

这对于“最终提交前固定流程”非常有帮助。

## 当前还没有完全满足，或者仍存在明显风险的地方

### 1. 与参考赛事测试方式的兼容层还不够完整

**当前问题：**

参考压缩包里的测试方式明显是围绕以下结构设计的：

- 根目录镜像名：`bdc2026`
- `docker compose -p test up -d`
- 容器内运行命令：`/bin/bash /app/data/run.sh`
- 输出结果从 `./test/output/result.csv` 拿

对应参考文件：

- `_case_zip/.../docker-compose.yml`
- `_case_zip/.../test/test.py`
- `_case_zip/.../test/test_windows.py`

而你的当前仓库：

- 没有仓库根目录的 `docker-compose.yml`
- 没有 `app/data/run.sh`
- Docker 默认入口是 [app/docker_rehearsal.sh](D:/Desktop/股票分析预测代码/app/docker_rehearsal.sh:1)，不是 `/app/data/run.sh`

这意味着：

- **你自己的 Docker 能构建，不等于一定能无缝通过参考测试脚本。**
- 如果赛事最终评测流程和参考压缩包接近，你现在这一点仍然有风险。

**具体要修改完善：**

1. 在 `app/data/` 下新增一个正式赛事入口脚本，例如 `app/data/run.sh`。
2. 让它只做赛事真正需要的流程，例如：
   - 初始化
   - 训练
   - 推理
   - 输出 `/app/output/result.csv`
3. 将 [Dockerfile](D:/Desktop/股票分析预测代码/Dockerfile:1) 的默认 `CMD` 改成更贴近赛事入口的脚本，而不是当前偏研发彩排的 `docker_rehearsal.sh`。
4. 在仓库根目录补一个本地兼容用的 `docker-compose.yml`，按参考压缩包方式挂载：
   - `./app/data -> /app/data`
   - `./test/output -> /app/output`
   - `./app/temp -> /app/temp`

### 2. Docker 默认入口更偏“研发彩排”，不够像正式提交入口

**当前问题：**

[Dockerfile](D:/Desktop/股票分析预测代码/Dockerfile:1) 当前的默认命令是：

- `CMD ["/bin/bash", "/app/docker_rehearsal.sh"]`

而 [app/docker_rehearsal.sh](D:/Desktop/股票分析预测代码/app/docker_rehearsal.sh:1) 会做：

- `init`
- `train`
- `test`
- `result_validator`
- `pre_submit_check`
- `cat result.csv`

这对本地彩排很好，但对正式评测镜像来说有两个风险：

- 启动逻辑偏重，做了很多“不是评测必需”的事
- 赛事如果只关心按约定入口生成 `result.csv`，你这个默认行为可能过于复杂

**具体要修改完善：**

1. 把容器入口拆成两类：
   - `run_submission.sh`：正式评测入口，只做必要动作
   - `docker_rehearsal.sh`：本地完整彩排入口
2. Docker 默认 `CMD` 建议切到正式入口脚本。
3. 把 `result_validator.py` 和 `pre_submit_check.py` 留给本地彩排脚本，不必强绑在正式容器启动路径里。

### 3. 当前镜像内容明显偏重，实验产物过多

**当前问题：**

当前 `app/` 目录文件总量很大，包含大量实验结果、回测图、搜索中间文件、历史模型与研究材料。仅 `app/` 下文件总体积就已经比较高，`app/model/` 中还有大量实验目录。

这会带来几个问题：

- Docker 构建慢
- 导出的 `.tar` 镜像更大
- 上传与加载更慢
- 评测启动时间和磁盘占用都更高

**具体要修改完善：**

1. 增加 `.dockerignore`，至少排除：
   - `app/temp/`
   - `app/output/`
   - `app/docs/experiment_notes.md`
   - `app/model/report_materials/`
   - 各类 `*_experiment/`
   - 各类 `*_search/`
   - `__pycache__/`
2. 准备一份“正式提交镜像只复制必要文件”的 Docker 打包策略：
   - 必需源码
   - 必需模型文件
   - 必需配置文件
   - 必需数据模板与运行脚本
3. 最好将正式提交所需模型单独收拢到一个目录，例如：
   - `app/model/submission_artifacts/`

### 4. 当前仓库缺少“参考测试脚本同口径”的本地模拟验证

**当前问题：**

你现在已经能跑：

- `test.ps1`
- `freeze_submission.ps1`
- `pre_submit_check.py`

但还缺一层“像赛事方那样加载 tar 镜像并跑容器”的本地自动验证。

参考压缩包里实际上提供了这种思路：

- `_case_zip/.../test/test.py`
- `_case_zip/.../test/test_windows.py`

**具体要修改完善：**

1. 在你自己的仓库里补一个 `submission_rehearsal/` 或 `test_submission/` 目录。
2. 提供以下脚本：
   - 构建镜像
   - 导出 tar
   - 加载 tar
   - 启动容器
   - 检查 `result.csv`
3. 让“正式提交前验证”不只停留在源码级别，而是升级到“tar 级别彩排”。

### 5. `pre_submit_check.py` 的检查还不够“真实评测化”

**当前问题：**

[app/code/src/pre_submit_check.py](D:/Desktop/股票分析预测代码/app/code/src/pre_submit_check.py:1) 虽然已经有用，但它现在的检查仍偏静态：

- 检查必需文件是否存在
- 检查几个配置文件是否一致

它还没有继续验证更关键的东西，比如：

- `model_meta.json` 中引用的模型路径是否真的能在提交环境下解析成功
- 实际推理入口是否能加载那个模型
- Docker 容器内默认路径是否和元数据兼容

**具体要修改完善：**

1. 在 `pre_submit_check.py` 中新增“真实模型路径解析检查”：
   - 读取 `model_meta.json`
   - 解析 `model_path`
   - 确认目标文件真实存在
2. 补一条“最小推理冒烟检查”：
   - 只要能成功加载模型并跑到输出阶段即可
3. 检查容器内路径口径是否统一，避免本地绝对路径在 Docker 中失效。

### 6. 当前元数据里仍保留了本机绝对路径，正式提交不够干净

**当前问题：**

[app/model/model_meta.json](D:/Desktop/股票分析预测代码/app/model/model_meta.json:1) 里仍包含本机绝对路径，例如：

- `feature_path`
- `model_path`

虽然当前 [app/code/src/config.py](D:/Desktop/股票分析预测代码/app/code/src/config.py:1) 里的 `resolve_metadata_artifact_path()` 已经做了回退处理，不一定会直接报错，但这类本机绝对路径出现在正式提交元数据里，仍然不够规范。

**具体要修改完善：**

1. 训练完成写 `model_meta.json` 时，优先写相对路径而不是本机绝对路径。
2. 推荐把模型路径固定成类似：
   - `app/model/submission_artifacts/lstm_model.pt`
3. 让元数据在 Windows、本地 Docker、Linux 评测机上都无需依赖绝对路径推断。

### 7. Dockerfile 里还没有把所有正式脚本权限一次性处理完整

**当前问题：**

[Dockerfile](D:/Desktop/股票分析预测代码/Dockerfile:1) 当前只 `chmod` 了：

- `/app/init.sh`
- `/app/train.sh`
- `/app/test.sh`
- `/app/docker_rehearsal.sh`

但 README 中已经公开了：

- `freeze_submission.sh`

它现在没有被加入 `chmod +x` 列表。

**具体要修改完善：**

1. 在 Dockerfile 中把以下脚本一并授权：
   - `/app/freeze_submission.sh`
   - 后续如果新增 `run_submission.sh`，也一起加入

### 8. 当前 README 已经较清晰，但还可以进一步向“赛事交付说明”收口

**当前问题：**

[app/readme.md](D:/Desktop/股票分析预测代码/app/readme.md:1) 现在已经比以前规整很多，但从赛事交付角度看，还可以再更简一点，尤其是：

- 把“正式提交主路径”和“研究/实验材料”分得更硬
- 把 Docker 提交操作再写得更接近最终动作

**具体要修改完善：**

1. 在 README 中补充：
   - 镜像构建命令
   - 镜像导出 tar 命令
   - 推荐镜像名
2. 增加“最终提交动作”小节，固定成：
   1. `freeze_submission`
   2. `docker build`
   3. `docker save`
   4. tar 彩排验证

## 当前最值得优先做的修改顺序

如果目标是尽快把“基本可用”推进到“更像正式可交付版本”，建议按下面顺序改：

1. **补赛事兼容入口**
   - 新增 `app/data/run.sh`
   - 新增正式提交容器入口脚本
   - 调整 Docker 默认 `CMD`

2. **瘦身提交镜像**
   - 增加 `.dockerignore`
   - 收拢正式提交模型与配置
   - 排除实验目录和中间文件

3. **补 tar 级别彩排**
   - 构建镜像
   - 导出 tar
   - 本地按参考测试方式加载并运行

4. **加强静态检查**
   - `pre_submit_check.py` 增加模型路径与最小推理检查
   - 元数据改为相对路径

## 最终判断

当前程序**已经不是“不符合要求”的状态**，相反，它已经具备了较强的正式提交基础：

- 能训练
- 能推理
- 能生成合法 `result.csv`
- 能做一致性检查
- 能打 Docker

但如果问题是：

> “现在能不能非常有把握地说，已经完全符合所有代码要求？”

我的判断是：

> **还不建议这么说。**

更稳妥的说法应该是：

> **你现在已经满足大部分核心要求，但还需要把 Docker 入口、赛事兼容运行方式、镜像瘦身、tar 级别彩排这几件事补齐，才更接近真正意义上的‘完全符合提交规范’。**
