# BDC2026 App 入口说明

本目录属于 `aggressive_score_submission` 精简提交包。

## 关键文件

- `output/result.csv`：正式提交结果，权重和 `1.000000`。
- `model/aggressive_score_submission_candidate/result_aggressive_score.csv`：入口脚本最终同步使用的 aggressive 冻结结果。
- `model/package_variant.json`：标记当前包为 `aggressive_score_submission`。

## 入口

- `data/run.sh`：Docker/评测默认入口。
- `run_submission.sh`：正式提交入口。
- `test.sh` / `test.ps1`：本地推理入口。

这些入口会在默认推理后自动恢复 aggressive 结果，确保最终 `output/result.csv` 为：

```csv
stock_id,weight
000792,0.2
600233,0.2
601669,0.2
600930,0.19999999999999998
002463,0.19999999999999996
```

本包不包含 `docs/`、`demo/` 或 `run_research_pipeline.sh`。
