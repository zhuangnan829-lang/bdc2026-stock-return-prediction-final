# 程序完善完成复盘与后续改进建议

生成日期：2026-05-24

本文档基于 `程序完善问题与解决方案清单.docx`、当前仓库代码、测试结果、模型产物和交付文档生成，用于总结本轮 30 项完善后程序实际提升在哪里，以及后续仍值得继续打磨的方向。

## 1. 总体结论

本轮完善已经把项目从“能跑出结果的研究代码”推进到“有配置来源、结果校验、自动测试、Docker 彩排、风控分析、候选模型对比和交付文档的完整工程闭环”。

最明显的提升不在单个模型结构，而在工程可信度：现在项目能够解释为什么选择 `LSTM sl20`，能证明正式输出如何生成，能用测试和校验脚本发现格式、配置、文件缺失、数据泄漏和 Docker 不一致问题，也能把后续模型增强放进统一比较框架，而不是凭感觉替换主线。

当前仍建议继续保留 `LSTM sl20 + base_alpha_v3_rs_crowding_mini4 + risk_adjusted + pred` 作为正式主线。`sl60`、`Transformer-lite`、LightGBM、XGBoost、Momentum 和 rank blend 已经进入候选或对照体系，但尚未在收益、稳定性、回撤、换手和复现上同时超过主线。

## 2. 本次实际提升

### 2.1 配置和提交链路更稳

原问题是正式参数分散在 shell、PowerShell、JSON、模型元信息和文档中，容易参数漂移。现在已经形成以配置文件为核心的入口体系：

- `app/model/default_submission_config.json`
- `app/model/best_config.json`
- `app/model/model_meta.json`
- `app/code/src/load_submission_config.py`
- `app/code/src/compare_config_consistency.py`
- `app/code/src/sync_submission_config.py`
- `app/run_submission.sh`
- `app/freeze_submission.sh`

本次验证结果：

```text
[compare_config_consistency] checks=45 failed=0
[compare_config_consistency] all checks passed
```

这说明正式配置之间已经可以被程序化检查，不再完全依赖人工核对。

### 2.2 提交结果有强制校验

现在正式输出不只是生成 `result.csv`，还会通过校验脚本检查格式、股票数量、权重和和必要文件：

- `app/code/src/result_validator.py`
- `app/code/src/pre_submit_check.py`
- `app/code/src/check_required_files.py`

本次验证结果：

```text
[pre_submit_check] required_files_ok=8
[pre_submit_check] result_ok rows=5 weight_sum=0.900000 encoding=utf-8
[pre_submit_check] config_consistency_ok=45
[pre_submit_check] all checks passed
```

这类检查能显著降低最后提交时因为列名、权重、文件缺失或配置不一致导致失败的风险。

### 2.3 自动化测试体系已建立

原来清单中的问题是缺少标准测试目录和 pytest。现在已经有 `tests/`，覆盖结果校验、回测、配置、特征、Docker 对比、CLI 入口、实验工具和文件检查等关键环节。

本次实际运行结果：

```text
26 passed in 16.81s
```

当前测试文件包括：

- `tests/test_result_validator.py`
- `tests/test_backtest_basic.py`
- `tests/test_feature_pipeline_smoke.py`
- `tests/test_config_files.py`
- `tests/test_readme_paths.py`
- `tests/test_check_required_files.py`
- `tests/test_docker_consistency_compare.py`
- `tests/test_unified_cli_entrypoints.py`
- `tests/test_feature_drift_reports.py`
- `tests/test_feature_set_presets.py`
- `tests/test_experiment_utils.py`

这让后续继续调参、重构入口或扩展模型时有了基本安全网。

### 2.4 CI 和跨平台入口更完整

已经新增 GitHub Actions：

- `.github/workflows/basic-ci.yml`

CI 会安装依赖、运行 pytest，并检查关键文件存在。项目也补了统一 CLI 和多平台薄封装：

- `app/code/src/cli.py`
- `app/train.sh`
- `app/test.sh`
- `app/train.ps1`
- `app/test.ps1`
- `app/run_train.bat`
- `app/run_test.bat`
- `app/run_freeze_submission.bat`

这比之前多套脚本各自维护参数更可控。

### 2.5 Docker 复现从“口头说明”变成了证据文件

当前已有 Docker 彩排和本机/Docker 对比证据：

- `Dockerfile`
- `docker-compose.yml`
- `app/docker_rehearsal.sh`
- `app/code/src/compare_local_docker_result.py`
- `app/model/docker_consistency_check.md`
- `test/output/result.csv`

历史记录显示 Docker 一致性曾经通过：

```text
status: PASS
local_result_md5: 9ce934cdfba9fa2162548d536aa68e7a
docker_result_md5: 9ce934cdfba9fa2162548d536aa68e7a
```

这对比赛交付很关键，因为评测环境通常更接近容器入口而不是本机交互运行。

### 2.6 风控和稳定性分析明显增强

清单中提到的单票集中、pred 权重激进、换手偏高、成本敏感等问题，现在已经有对应实验产物：

- `app/code/src/analyze_position_concentration.py`
- `app/code/src/search_weight_cap.py`
- `app/code/src/search_weight_blend.py`
- `app/code/src/turnover_stress_test.py`
- `app/model/weight_cap_search/weight_cap_search_summary.csv`
- `app/model/weight_blend_search/weight_blend_summary.csv`
- `app/model/turnover_stress_test/turnover_stress_summary.csv`
- `app/output/slice_concentration_summary.csv`
- `app/output/slice_concentration_detail.csv`

关键观察：

| 方案 | 成本后累计收益 | Sharpe | 最大回撤 | 平均换手 | 说明 |
|---|---:|---:|---:|---:|---|
| `max_single_weight_0.20` | 1.165010 | 4.002179 | -0.090852 | 0.956291 | 收益接近主线，集中度仍较高。 |
| `max_single_weight_0.18` | 1.085936 | 3.977455 | -0.087064 | 0.931137 | 降低集中度，收益略降。 |
| `max_single_weight_0.16` | 0.999800 | 3.930683 | -0.082412 | 0.898850 | 回撤和换手更低，但收益进一步下降。 |

这说明程序现在能把“冲分”和“稳健”拆开讨论，而不是只看最终单次得分。

### 2.7 模型增强不再盲目替换主线

已经补齐候选分支和同协议比较：

- `app/code/src/run_sequence_length_search.py`
- `app/code/src/train_transformer_lite.py`
- `app/code/src/transformer_lite_utils.py`
- `app/code/src/rank_blend.py`
- `app/model/sequence_length_search/sl20_sl40_sl60_compare.csv`
- `app/model/transformer_lite_sl60_compare.csv`
- `app/model/rank_blend/blend_summary.csv`
- `app/docs/model_selection_rationale.md`

当前核心对比：

| 分支 | RankIC | Top5 | Sharpe | 最大回撤 | 切片得分 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `lstm_sl20` | 0.027982 | 0.007502 | 3.977455 | -0.087064 | 0.031491 | 保留主线 |
| `lstm_sl60` | 0.039856 | 0.010208 | 2.481363 | -0.099273 | 0.002946 | 只作候选 |
| `transformer_lite_sl60` | -0.028891 | 0.004924 | 0.792688 | -0.121362 | -0.022891 | 不替换 |

这比“看到 Transformer 或 sl60 就换模型”稳很多。当前已经有明确原则：只有新分支在切片得分、RankIC、Top5、Sharpe、回撤、换手和 Docker 复现上同时不差，才替换。

### 2.8 数据泄漏和实验追溯能力增强

新增了数据防泄漏检查：

- `app/code/src/check_data_leakage.py`
- `app/model/data_leakage_check_report.md`

本次验证结果：

```text
[check_data_leakage] overall=PASS
[PASS] feature date <= prediction date
[PASS] training label horizon is forward only
[PASS] label fields excluded from model features
[PASS] walk-forward train/validation order
[PASS] walk-forward windows reproducible from feature dates
```

同时，实验索引和排行榜已经形成：

- `app/docs/experiment_result_index.md`
- `app/model/experiment_leaderboard.csv`

当前 `experiment_leaderboard.csv` 已汇总 4134 行实验记录。这对复盘和答辩很加分。

### 2.9 README 和交付文档更像正式项目

根 README 已经把正式使用入口前置，覆盖：

- 安装依赖
- Research 复现
- Submission 正式提交
- Freeze 冻结
- Docker 彩排
- Streamlit Demo
- 正式配置来源
- 校验方式
- 常见问题

同时新增或完善了多份交付文档：

- `app/docs/final_delivery_checklist.md`
- `app/docs/experiment_result_index.md`
- `app/docs/reproducibility_guide.md`
- `app/docs/environment_snapshot.md`
- `app/docs/model_selection_rationale.md`
- `app/docs/report_supplement_tables.md`

这让项目更容易被评审、队友或未来的自己快速接手。

## 3. 当前最值得继续改进的问题

### 3.1 最高优先级：重新冻结当前正式输出，消除快照与 result 不一致

当前发现一个交付口径风险：

- `app/output/result.csv` 当前内容为 5 只股票各 `0.18`，权重和 `0.900000`。
- `app/model/final_submission_snapshot.md` 仍记录旧的 pred 权重，权重和为 `1.000000`。
- `test/output/result.csv` 仍是 Docker 彩排旧结果，和当前 `app/output/result.csv` MD5 不一致。
- `app/model/docker_consistency_check.md` 记录的是旧彩排通过结果。

这说明当前程序校验能通过，但“当前工作区输出、冻结快照、Docker 彩排记录”不是同一个时间点的产物。

建议立刻做一次最终冻结彩排：

```bash
bash app/freeze_submission.sh
bash app/run_submission.sh
docker compose -p bdc2026_rehearsal up --abort-on-container-exit --force-recreate
python app/code/src/compare_local_docker_result.py
python app/code/src/pre_submit_check.py --root_dir . --result_path app/output/result.csv
```

完成后应确认：

- `app/output/result.csv`
- `app/model/final_submission_snapshot.md`
- `app/model/submission_artifacts/final_submission_snapshot.md`
- `test/output/result.csv`
- `app/model/docker_consistency_check.md`

五者口径一致。

### 3.2 明确最终提交到底用 `pred` 权重还是 `0.18` 权重上限版

当前配置文档仍写正式策略是 `pred`，但当前 `result.csv` 表现为等权上限式输出。两者都可能合理：

- `pred` 权重更像 aggressive 冲分版。
- `0.18` 单票上限版更像 robust 稳健版。

建议最终只保留一个“正式提交默认”，另一个作为备选配置写入 `app/model/configs/`。如果最终决定用 `0.18` 版，就应更新快照、README 和模型选择文档中的正式输出描述。

### 3.3 Docker 一致性检查应加入最后提交前强制门禁

目前已经有 `compare_local_docker_result.py` 和历史 PASS 记录，但从当前输出看，Docker 产物可能滞后于本机产物。建议在 `freeze_submission.sh` 或最终彩排流程中增加一条明确要求：

- 如果 `test/output/result.csv` 存在，则必须和 `app/output/result.csv` 内容一致。
- 如果 Docker 彩排未执行，则最终交付清单必须标记为“待彩排”。

这样能防止文档里写 Docker PASS，但实际当前提交文件已经变化。

### 3.4 CI 还可以加入配置一致性和提交前检查

当前 CI 已运行 pytest 和关键文件存在检查。下一步可以把轻量脚本也放进 CI：

```bash
python app/code/src/compare_config_consistency.py
python app/code/src/pre_submit_check.py --root_dir .
python app/code/src/check_data_leakage.py --app-root app
```

这样每次 push 不仅证明测试通过，还能证明配置和交付口径没有漂移。

### 3.5 Demo 和 PPT 仍需要最后人工彩排

`app/docs/final_delivery_checklist.md` 中仍有未勾选项，主要集中在答辩材料：

- PPT 模型清单与 `opening_report_alignment.md` 逐页对齐
- PPT 正式默认方案与 `final_submission_snapshot.md` 对齐
- PPT 图表引用路径逐页核对
- Demo 展示顺序与 `app/docs/demo_3min_main_flow.md` 完全一致
- 再启动一次 Demo 做最终展示彩排
- 核对 README、报告、PPT、Demo 口径完全一致

这些不是核心程序问题，但会直接影响最终展示质量。

### 3.6 实验排行榜已经很丰富，但可以增加“最终推荐列”

当前 `experiment_leaderboard.csv` 已经有 4134 行，说明实验覆盖非常充分。后续可以在排行榜中再加一层人工可读结论：

- `candidate_role`: mainline / aggressive / robust / baseline / rejected
- `reject_reason`: low_rankic / high_drawdown / high_turnover / low_slice_score / unstable_fold
- `delivery_ready`: true / false

这样答辩时可以更快解释“为什么这么多实验最后仍选择 sl20”。

### 3.7 特征集增强仍可继续，但不要为了数量扩特征

当前已经有中等特征集相关代码和特征漂移分析能力，但 `app/model/feature_set_search/feature_set_comparison.csv` 当前未发现。建议下一阶段继续做：

- `base_alpha_v4_medium` 与当前 20 特征同协议比较
- 特征漂移报告和特征重要性结合
- 只采用 walk-forward 和稳定性都提升的特征，不采用只提升训练集或单切片的特征

这条属于后续提升空间，不影响当前交付闭环。

## 4. 建议的后续优先级

| 优先级 | 建议事项 | 目标 |
|---|---|---|
| P0 | 重新执行最终冻结和 Docker 彩排 | 消除当前 result、快照、Docker 产物不一致。 |
| P0 | 明确最终默认是 aggressive 还是 robust | 避免正式配置说明和实际 result 权重不一致。 |
| P0 | 把配置一致性、pre-submit、Docker 对比加入最终门禁 | 防止最后一天改动造成漂移。 |
| P1 | 完成 Demo/PPT/README/快照逐项口径核对 | 提升答辩展示一致性。 |
| P1 | 给实验排行榜增加推荐状态和拒绝原因 | 让模型选择依据更直观。 |
| P2 | 继续做中等特征集和特征漂移联动分析 | 在不牺牲交付稳定性的前提下寻找增益。 |
| P2 | 将 Docker 彩排和结果对比自动化成单条命令 | 降低人工彩排成本。 |

## 5. 一句话总结

你这轮最大的提升是：项目已经不只是“模型能预测”，而是形成了“配置可追溯、结果可校验、实验可比较、风险可解释、Docker 可复现、文档可交付”的完整体系。下一步最该做的不是继续堆模型，而是完成最后一次冻结与 Docker 彩排，把当前输出、快照和交付文档对齐到同一个最终版本。

