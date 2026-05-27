## 当前程序相对赛题要求与代码规范的主要不足及改进方案

### 1. 默认推理口径与配置快照尚未完全统一
**不足说明：**  
当前实际生效的默认推理参数主要体现在 `app/test.sh` 和 `app/test.ps1` 中，但正式提交相关配置还同时分散在 `app/model/best_config.json`、`app/model/default_submission_config.json`、`app/model/model_meta.json` 等文件里。这样会出现“脚本实际跑的是一套、配置声明的是另一套”的风险，不利于提交一致性，也可能导致 `pre_submit_check.py` 无法通过。

**具体改进做法：**  
- 确定唯一正式默认方案，例如当前已经验证过的 `rp-60` 默认口径。  
- 将该方案同步写入：
  - `app/test.sh`
  - `app/test.ps1`
  - `app/model/best_config.json`
  - `app/model/default_submission_config.json`
  - `app/model/model_meta.json`
- 每次调整正式默认后，强制执行一次：
  - `python app/code/src/pre_submit_check.py --root_dir / --result_path app/output/result.csv`
- 后续将“实验候选参数”与“正式提交参数”分开存放，避免混用。

### 2. README 更偏内部研发说明，尚未完全收敛为提交文档
**不足说明：**  
当前 `app/readme.md` 已经能说明项目流程，但仍包含较多内部实验语境，例如“双目标搜索”“推荐冠军”“压缩包对比”等。这些内容对研发有帮助，但对评测方来说不一定是最清晰、最直接的交付说明。

**具体改进做法：**  
- 将 README 分成两类：
  - `app/readme.md`：正式提交说明，只保留交付所需内容
  - `app/docs/experiment_notes.md`：保留实验、对比、调参记录
- 正式 README 建议固定为以下结构：
  1. 项目目标
  2. 运行环境
  3. 输入数据说明
  4. 训练方式
  5. 推理方式
  6. 输出文件说明
  7. 提交前校验
  8. Docker 离线复现
- 删除或弱化内部研究口径描述，避免让交付文档看起来像实验日志。

### 3. Windows 侧运行仍依赖 `ExecutionPolicy Bypass`
**不足说明：**  
当前在 Windows 下直接运行 `.\app\test.ps1` 往往会被 PowerShell 执行策略拦截，需要通过 `powershell -ExecutionPolicy Bypass -File .\app\test.ps1` 才能执行。这虽然不影响功能，但会降低“开箱即用性”。

**具体改进做法：**  
- 在 README 中明确写出 Windows 标准运行方式，不再默认用户直接双击或直接执行 `.ps1`。
- 增加一个更直白的 Windows 启动包装，例如：
  - `app/run_test.bat`
  - `app/run_train.bat`
- 在 `.bat` 里统一调用：
  - `powershell -ExecutionPolicy Bypass -File .\app\test.ps1`
- 这样可以降低本地复现门槛，避免使用者被执行策略问题卡住。

### 4. 正式默认方案的“权威来源”还不够单一
**不足说明：**  
目前仓库里同时存在多种“看起来像默认”的来源，例如：
- `test.sh/test.ps1`
- `best_config.json`
- `default_submission_config.json`
- `model_meta.json`
- dual-objective 搜索推荐结果
- README 中的默认说明

这会让“到底哪一套才是正式提交版本”变得不够明确。

**具体改进做法：**  
- 指定唯一权威来源，例如：
  - `app/model/default_submission_config.json`
- 其他文件全部从这个权威来源自动同步或显式引用。
- 增加一个同步脚本，例如：
  - `app/code/src/sync_submission_config.py`
- 该脚本负责把正式默认写回：
  - `test.sh`
  - `test.ps1`
  - `best_config.json`
  - `model_meta.json`
- 这样可以把“配置漂移”风险降到最低。

### 5. 内部多期回测很完整，但与赛事公开评分口径仍需分层表达
**不足说明：**  
当前程序已经具备 `cumulative_return_after_cost`、`sharpe_after_cost`、`max_drawdown_after_cost`、`avg_turnover` 等完整回测指标，这是明显优点；但赛事方公开可直接对齐的核心仍然是最终 `result.csv` 的评分逻辑。若表达不清，容易让人误以为内部指标和官方评分是同一个东西。

**具体改进做法：**  
- 在文档中明确区分两类指标：
  - **官方评分口径**：单次 `result.csv` 的最终得分
  - **内部研究口径**：walk-forward、Sharpe、回撤、换手等
- 在 README 或报告中固定用两个章节分别说明：
  - “提交表现”
  - “研究评估”
- 避免直接把内部回测指标写成“官方收益率证明”，而是表述为“本地稳定性证据”。

### 6. 正式提交流程和实验流程尚未彻底分离
**不足说明：**  
当前仓库里既有正式训练/推理脚本，也有大量实验脚本，例如双目标搜索、候选推荐、压缩包对比等。研发角度这是好事，但如果没有边界，容易增加交付复杂度，也可能让使用者误操作到实验脚本。

**具体改进做法：**  
- 将流程分层：
  - 正式提交路径：`init -> train -> test -> validate -> pre_submit_check`
  - 实验研究路径：搜索、对比、调参、候选分析
- 建议目录上进一步区分：
  - `app/code/src/submission/`
  - `app/code/src/research/`
- 或至少在文件命名层面区分：
  - `run_submission_*`
  - `run_research_*`
- 在 README 中只保留正式提交流程，把实验脚本放到附录或单独文档。

### 7. 中文文档与终端显示存在编码可读性问题
**不足说明：**  
当前部分中文文档在终端 `Get-Content` 查看时会出现乱码。虽然文件本身未必损坏，但会影响本地阅读体验，也会降低文档专业度。

**具体改进做法：**  
- 统一所有 Markdown、CSV、JSON 文本文件保存为 UTF-8。
- 对 README、报告类文件尽量使用 UTF-8 无 BOM 或明确统一编码策略。
- 在 PowerShell 环境中补充编码说明，必要时先执行：
  - `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`
- 重新检查关键文档：
  - `app/readme.md`
  - 各类 `report.md`
  - 配置和结果说明文件
- 保证在常见 Windows 终端和编辑器里都能直接正常阅读。

### 8. 当前缺少一轮完整的“提交收口”动作
**不足说明：**  
目前模型、执行层、搜索逻辑已经迭代到比较强的状态，但仓库整体还停留在“研发持续演进”阶段，没有彻底完成“正式提交冻结”。

**具体改进做法：**  
- 建议进行一次完整的提交冻结：
  1. 选定唯一正式默认方案
  2. 同步全部配置文件
  3. 重跑 `.\app\test.ps1`
  4. 生成最终 `result.csv`
  5. 执行 `result_validator.py`
  6. 执行 `pre_submit_check.py`
  7. 更新正式 README
  8. 记录最终提交快照
- 额外产出一个冻结说明文件，例如：
  - `app/model/final_submission_snapshot.md`
- 用来记录本次正式提交所对应的：
  - 模型
  - 特征集
  - 执行参数
  - 校验结果
  - 最终输出文件
