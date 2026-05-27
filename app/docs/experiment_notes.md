# 实验与研究说明

本文件用于记录正式提交路径之外的研究工作，便于后续继续优化，但不作为正式提交主流程说明。

## 当前研究方向

- 压缩包程序单切片对比
- LSTM 执行层参数搜索
- dual-objective 搜索
- 候选池大小、风险过滤、换手控制调优
- 本地 walk-forward 稳定性分析

## 当前主要研究脚本

- `app/code/src/build_case_program_comparison.py`
- `app/code/src/run_lstm_execution_search.py`
- `app/code/src/run_lstm_dual_objective_search.py`
- `app/code/src/run_model_comparison.py`
- `app/code/src/run_alpha_rs_crowding_mini4_experiment.py`
- `app/code/src/run_alpha_v3_ablation.py`

## 当前研究产物

- `app/model/case_program_comparison/`
- `app/model/lstm_dual_objective_search/`
- `app/model/report_materials/`

## 说明

- 正式提交时只需关注 `app/readme.md` 中列出的主流程。
- 本文件中的脚本和产物主要服务于调参、对比和复盘。
