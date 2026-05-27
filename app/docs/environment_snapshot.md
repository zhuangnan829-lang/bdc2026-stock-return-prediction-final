# 环境快照

本文档记录当前项目在本地整理与交付阶段的运行环境、关键入口与 Docker 离线彩排结果，便于复现训练、推理与正式提交链路。

## 1. 本地 Python 环境

- Python：`3.12.7`
- 平台：`Windows-11-10.0.26200-SP0`

## 2. requirements.txt 快照

当前 `requirements.txt` 包含：

- `pandas==2.2.3`
- `numpy==2.1.3`
- `scikit-learn==1.5.2`
- `joblib==1.4.2`
- `lightgbm==4.5.0`
- `matplotlib==3.9.2`
- `torch==2.11.0`
- `streamlit==1.37.1`

## 3. Docker 环境快照

Dockerfile 当前基础镜像：

- `python:3.12-slim`

Dockerfile 额外安装：

- `libgomp1`

默认容器工作目录：

- `/app`

默认 CMD：

- `/bin/bash /app/data/run.sh`

## 4. Docker 离线彩排记录

- 彩排日期：`2026-05-18`
- 执行链路：`docker build -t bdc2026 .` -> `docker save -o bdc2026_rehearsal.tar bdc2026:latest` -> `docker rmi bdc2026:latest` -> `docker load -i bdc2026_rehearsal.tar` -> `docker compose -p bdc2026_rehearsal up --abort-on-container-exit --force-recreate`
- 导出镜像 tar：`bdc2026_rehearsal.tar`
- tar SHA256：`7c7e1f003d22d01b21d2ad15674ef4e24d1fe6e72c1ec225656c2d4909310629`
- 容器输出目录：`test/output/`
- 复现产物：
  - `test/output/result.csv`
  - `test/output/predict_scores.csv`
  - `test/output/debug_candidates.csv`
- 结果校验：
  - `app/output/result.csv` SHA256：`f447d01eaf89a41fa9b85de4bc4c712c2501d63dbf2619b03d7afe9710e1da47`
  - `test/output/result.csv` SHA256：`cac147b0f764727569469ec107514e3976896801a7480fc3ee0a6b85de4946da`
  - `app/output/predict_scores.csv` SHA256：`b83db127292662aad971efffe81ea7b4829a32e9403661309acaf1ea8bcc7eee`
  - `test/output/predict_scores.csv` SHA256：`eb25c6d541705bd48c6657f533d7370872cf4c7b28b1c06952ae44a06d17372d`
  - `app/output/debug_candidates.csv` SHA256：`df7736a4fe5ffe4017953fcd37cc74e100e4780f47b439a7efd4977561d86955`
  - `test/output/debug_candidates.csv` SHA256：`F6FDDA2472650C333EC5DBCDBE973F079F49A729CA94E16477C52DA4EBFA05B0`
  - `result.csv`、`predict_scores.csv`、`debug_candidates.csv` 三个文件按文本内容逐行一致
  - SHA256 差异仅来自宿主机 `CRLF` 与容器 `LF` 的换行格式不同

## 5. 关键脚本快照

当前核心链路脚本：

- `app/init.sh`
- `app/train.sh`
- `app/test.sh`
- `app/run_research_pipeline.sh`
- `app/run_submission.sh`
- `app/freeze_submission.sh`
- `app/demo/run_demo.ps1`
- `app/demo/run_demo.sh`

## 6. 关键配置快照

当前正式默认方案相关快照文件：

- `app/model/best_config.json`
- `app/model/default_submission_config.json`
- `app/model/model_meta.json`
- `app/model/final_submission_snapshot.md`
- `app/model/submission_artifacts/manifest.json`

## 7. 关键输出目录快照

### 7.1 研究/模型产物

- `app/model/`

### 7.2 正式推理输出

- `app/output/result.csv`
- `app/output/predict_scores.csv`
- `app/output/debug_candidates.csv`

### 7.3 Docker 彩排输出

- `test/output/result.csv`
- `test/output/predict_scores.csv`
- `test/output/debug_candidates.csv`

### 7.4 Demo 与报告图件

- `app/docs/report_supplement_assets/`
- `app/docs/figures/midterm/`

## 8. 说明

- 本文档是项目整理与交付阶段的环境快照，不等同于比赛官方评测环境。
- 若后续切换 Python、依赖版本或 Docker 基础镜像，建议同步更新本文件。
- 本次 Docker 离线彩排过程中修复了两个交付链路问题：一是镜像内 shell 脚本的 `CRLF` 换行兼容性，二是 LSTM 推理阶段缺少 `train_features.csv` 历史特征上下文时的自动补全逻辑。
