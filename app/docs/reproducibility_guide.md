# 复现指南

本文档用于说明本项目当前三条主要链路的复现方法：

1. 研究链路：训练、walk-forward、回测、诊断
2. 正式提交链路：冻结配置、推理、结果校验
3. Demo 链路：答辩展示页面

## 1. 目录与入口约定

主要入口脚本如下：

- 研究链路：`app/run_research_pipeline.sh`
- 正式提交链路：`app/run_submission.sh`
- 冻结提交链路：`app/freeze_submission.sh`
- Demo 链路：`app/demo/run_demo.ps1`、`app/demo/run_demo.sh`

主要输入目录：

- `app/data/`
- `app/temp/`

主要输出目录：

- `app/model/`
- `app/output/`
- `app/docs/report_supplement_assets/`

## 2. 运行前准备

### 2.1 依赖安装

建议先安装：

```bash
pip install -r requirements.txt
```

### 2.2 关键输入文件

至少应确认以下文件存在：

- `app/data/train.csv`
- `app/data/test.csv`
- `app/data/stock_data.csv`
- `app/data/hs300_stock_list.csv`

### 2.3 Windows 控制台中文显示

如果 PowerShell 出现中文乱码，可先执行：

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

## 3. 研究链路复现

研究链路入口：

```bash
bash /app/run_research_pipeline.sh
```

Windows 本地通常在项目根目录执行：

```powershell
bash ./app/run_research_pipeline.sh
```

### 3.1 研究链路会做什么

`run_research_pipeline.sh` 默认依次执行：

1. `train.sh`
2. `test.sh`
3. `result_validator.py`
4. `backtest.py`
5. fold diagnostics 导出

### 3.2 主要产物

研究链路完成后，重点查看：

- `app/model/lstm_model.pt`
- `app/model/model_meta.json`
- `app/model/walk_forward_predictions.csv`
- `app/model/walk_forward_metrics.csv`
- `app/model/backtest_summary.csv`
- `app/model/backtest_report.md`
- `app/model/fold_diagnostics.csv`
- `app/model/fold_daily_diagnostics.csv`
- `app/model/fold_1_predictions.csv`
- `app/output/result.csv`

### 3.3 可控开关

可通过环境变量控制后续步骤是否执行：

- `RUN_BACKTEST=0`：跳过本地回测
- `RUN_DIAGNOSTICS=0`：跳过折次诊断
- `RUN_RESULT_VALIDATION=0`：跳过结果格式校验

## 4. 正式提交链路复现

正式提交入口：

```bash
bash /app/run_submission.sh
```

这条链路的默认设计是：

- `RUN_TRAIN=0`
- `RUN_VALIDATION=0`

也就是默认**不重新训练**，而是使用冻结模型直接做正式推理。

### 4.1 主要输出

- `app/output/result.csv`
- `app/output/predict_scores.csv`
- `app/output/debug_candidates.csv`

### 4.2 适用场景

- 容器内正式提交
- 本地模拟正式推理
- 不希望训练结果漂移时的固定口径复现

## 5. 冻结提交链路复现

冻结提交流程入口：

```bash
bash /app/freeze_submission.sh
```

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File .\app\freeze_submission.ps1
```

### 5.1 这条链路会做什么

`freeze_submission.sh` 默认依次执行：

1. `sync_submission_config.py`
2. `test.sh`
3. `result_validator.py`
4. `pre_submit_check.py`
5. `build_case_program_comparison.py`

### 5.2 重点产物

- `app/output/result.csv`
- `app/model/best_config.json`
- `app/model/default_submission_config.json`
- `app/model/final_submission_snapshot.md`
- `app/model/submission_artifacts/`

如果要做“答辩前最后一轮彩排”，优先跑这条链路。

## 6. Demo 链路复现

### 6.1 Windows 启动

在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\app\demo\run_demo.ps1
```

### 6.2 Linux/macOS 启动

```bash
bash /app/demo/run_demo.sh
```

### 6.3 Demo 端口说明

Windows 版 `run_demo.ps1` 会自动尝试以下端口：

- `8501`
- `8502`
- `8503`
- `8510`
- `8601`

也就是说：

- Demo 端口**不保证固定是 8501**
- 如果 `8501` 被占用，会自动切换到下一个可用端口
- 终端会打印最终 URL，例如 `http://127.0.0.1:8502`

### 6.4 Demo 读取哪些结果文件

当前 Demo 主要读取：

- `app/model/best_config.json`
- `app/model/formal_model_comparison/formal_model_comparison.csv`
- `app/model/market_regime_analysis/fold_stage_performance.csv`
- `app/model/backtest_summary.csv`
- `app/model/fold_1_predictions.csv`
- `app/model/fold1_short_term_ticket_summary.csv`
- `app/model/fold1_short_term_ticket_diagnostics.csv`
- `app/docs/report_supplement_assets/formal_model_comparison_chart.png`
- `app/docs/figures/midterm/fig2_fold_rankic.png`
- `app/docs/figures/midterm/fig3_equity_curve.png`
- `app/docs/figures/midterm/fig4_drawdown_curve.png`
- `app/docs/figures/midterm/fig6_ticket_diagnostics.png`

### 6.5 Demo 启动后的查看方式

优先看终端打印出来的 URL，例如：

```text
[run_demo.ps1] url: http://127.0.0.1:8502
```

然后在浏览器打开该地址。

## 7. Docker 复现

### 7.1 构建镜像

```bash
docker build -t bdc2026 .
```

### 7.2 本地编排

```bash
docker compose up
```

当前 `docker-compose.yml` 主要挂载：

- `./app/data -> /app/data`
- `./test/output -> /app/output`
- `./app/temp -> /app/temp`

### 7.3 容器默认入口

Docker 默认入口为：

- `/app/data/run.sh`

该入口最终会调用：

- `app/run_submission.sh`

## 8. 常见问题

### 8.1 Demo 打不开

优先检查：

1. 终端里最终打印的是哪个端口
2. 当前端口是否被本机安全软件拦截
3. 是否误以为固定是 `8501`

### 8.2 只想看正式结果，不想重新训练

优先使用：

```bash
bash /app/run_submission.sh
```

或：

```bash
bash /app/freeze_submission.sh
```

### 8.3 想要完整实验闭环

优先使用：

```bash
bash /app/run_research_pipeline.sh
```

## 9. 推荐使用顺序

如果是答辩前彩排，建议顺序：

1. `bash /app/freeze_submission.sh`
2. 确认 `app/output/result.csv`
3. 启动 `app/demo/run_demo.ps1`
4. 打开 Demo 页面讲解结果
