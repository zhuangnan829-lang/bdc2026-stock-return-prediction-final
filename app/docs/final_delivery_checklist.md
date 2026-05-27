# 最终交付清单

本文档用于提交前逐项自查。我已将当前已经核实通过的项目打勾，尚未完成最终复核的项目继续保留未勾选状态。

## 1. 代码交付

- [x] `app/code/src/` 核心源码存在
- [x] `app/init.sh`、`app/train.sh`、`app/test.sh` 存在
- [x] `app/run_research_pipeline.sh` 存在
- [x] `app/run_submission.sh` 存在
- [x] `app/freeze_submission.sh` 存在
- [x] Windows 对应脚本存在：`*.ps1`、`*.bat`
- [x] `requirements.txt` 已记录运行依赖
- [x] `Dockerfile`、`docker-compose.yml` 存在且入口口径一致

## 2. 模型与配置交付

- [x] `app/model/lstm_model.pt`
- [x] `app/model/model_meta.json`
- [x] `app/model/best_config.json`
- [x] `app/model/default_submission_config.json`
- [x] `app/model/final_submission_snapshot.md`
- [x] `app/model/submission_artifacts/` 目录存在且内容完整
- [x] `app/model/submission_artifacts/manifest.json` 存在

## 3. 结果文件交付

- [x] `app/output/result.csv`
- [x] `app/output/predict_scores.csv`
- [x] `app/output/debug_candidates.csv`
- [x] `app/model/formal_model_comparison/formal_model_comparison.csv`
- [x] `app/model/backtest_summary.csv`
- [x] 市场阶段分析图表素材已保留在 `app/docs/report_supplement_assets/market_regime_analysis_chart.png`
- [x] `app/model/fold_1_predictions.csv`
- [x] `app/model/fold1_short_term_ticket_summary.csv`
- [x] `app/model/fold1_short_term_ticket_diagnostics.csv`

## 4. 文档交付

- [x] `app/readme.md`
- [x] `app/docs/opening_report_alignment.md`
- [x] `app/docs/reproducibility_guide.md`
- [x] `app/docs/environment_snapshot.md`
- [x] `app/docs/experiment_result_index.md`
- [x] `app/docs/report_figure_table_index.md`
- [x] `app/docs/unimplemented_model_applicability.md`
- [x] `app/docs/final_presentation_alignment.md`
- [x] `app/docs/final_delivery_checklist.md`
- [x] `app/docs/结题报告_沪深300股票收益预测与组合推荐.docx`
- [x] 中期/答辩相关展示口径已完成最终统一复核

## 5. 报告图表交付

- [x] `app/docs/report_supplement_assets/formal_model_comparison_chart.png`
- [x] `app/docs/report_supplement_assets/market_regime_analysis_chart.png`
- [x] `app/docs/report_supplement_assets/demo_flowchart.png`
- [x] `app/docs/report_supplement_assets/demo_key_commands.png`
- [x] `app/docs/report_supplement_assets/demo_result_files.png`
- [x] `app/docs/report_supplement_tables.md`
- [x] `app/docs/figures/midterm/` 目录存在且包含当前图像文件

## 6. Demo 交付

- [x] `app/demo/streamlit_app.py`
- [x] `app/demo/run_demo.ps1`
- [x] `app/demo/run_demo.sh`
- [x] 本机已验证 Demo 可以启动
- [x] 已确认 Demo 所需 csv/png 路径存在
- [x] 已确认端口自动切换逻辑可用

## 7. 环境说明交付

- [x] `requirements.txt` 可作为基础安装入口
- [x] `Dockerfile` 已实际完成一次本地构建验证
- [x] `docker-compose.yml` 已实际完成一次本地彩排验证
- [x] `app/docs/environment_snapshot.md` 已更新
- [x] README 与环境文档已说明本地环境入口

## 8. Docker 说明交付

- [x] 已确认默认入口 `/app/data/run.sh`
- [x] 已确认该入口会调用 `app/run_submission.sh`
- [x] 已确认 volume 挂载路径与文档说明一致
- [x] 已将 `docker build -t bdc2026 .` 写入文档
- [x] 已在文档中补充 `docker save` 说明
- [x] 已完成 `docker save -> docker load -> docker compose up` 的 tar 级离线彩排
- [x] 已核对容器复现结果与仓库基准结果内容一致

## 9. PPT / 答辩材料交付

说明：当前仓库未包含正式 `.pptx` 文件；本节“已对齐”指 PPT 制作口径和逐页核对稿已在 `app/docs/final_presentation_alignment.md` 中固定。

- [x] PPT 模型清单口径已与 `opening_report_alignment.md` / `final_presentation_alignment.md` 对齐
- [x] 开题承诺但未落地的 `ARIMA / TFT / N-HiTS / TSFM / Qlib / FinRL` 已整理为调研与适用性说明
- [x] PPT 正式默认方案口径已与 `final_submission_snapshot.md` 对齐
- [x] PPT 图表引用路径已在 `final_presentation_alignment.md` 中统一核对
- [x] Demo 展示顺序已与 `app/docs/demo_3min_main_flow.md` 完全一致
- [x] 已准备“为什么选 LSTM sl20”的标准回答依据
- [x] 正式答辩 PPT 已归档：`app/docs/沪深300股票收益预测与组合推荐_学术研究汇报_答辩终版.pptx`

## 10. 提交前最后检查

- [x] 重新执行一次 `app/freeze_submission.sh`
- [x] 检查 `app/output/result.csv` 最新格式与权重和
- [x] 检查 `pre_submit_check.py` 最新结果
- [ ] 再启动一次 Demo 并做最终展示彩排
- [x] 核对 README、报告、PPT、Demo 口径完全一致
- [x] 已完成一次 Docker 离线回装彩排并记录结果

## 11. 推荐提交前彩排顺序

1. 执行 `bash /app/freeze_submission.sh`
2. 检查 `app/output/result.csv`
3. 打开 `app/model/final_submission_snapshot.md`
4. 启动 `app/demo/run_demo.ps1`
5. 用 Demo 走一遍答辩讲解顺序
