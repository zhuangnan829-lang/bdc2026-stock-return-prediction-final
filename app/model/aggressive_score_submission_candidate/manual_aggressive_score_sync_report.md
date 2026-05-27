# Manual Aggressive Score Sync Report

本报告记录一次人工确认版 aggressive score 临时同步。脚本已备份当前 HV rerank 默认结果，再把 aggressive score 候选写入 `app/output/result.csv`。

## Decision

- sync status: `active`
- default config sync: `not changed`
- result sync: `app/output/result.csv has been replaced by aggressive score candidate`
- rollback: copy the backed-up result file back to `app/output/result.csv`, or run the restore command below.

## Files

- backup directory: `D:\Desktop\股票分析预测代码\app\model\manual_switch_backups\aggressive_score_manual_sync_20260526_135842`
- candidate result: `D:\Desktop\股票分析预测代码\app\model\aggressive_score_submission_candidate\result_aggressive_score.csv`
- output result: `D:\Desktop\股票分析预测代码\app\output\result.csv`
- backup result sha256: `6d0946ef4d164f4fd2d5ad21115155ff82592febbfa91d0a1802e1be8c764cb1`
- current output sha256: `8b0b54185d6c9922491e41f0bea7dab978249ce991c5af0e38352fa08a8035fb`

## Rollback Command

```powershell
python app/code/src/manual_sync_aggressive_score_candidate.py --restore_backup "D:\Desktop\股票分析预测代码\app\model\manual_switch_backups\aggressive_score_manual_sync_20260526_135842"
```

## Before Sync: HV Rerank Result

| stock_id | weight |
|---|---:|
| `300316` | 0.180000 |
| `600115` | 0.180000 |
| `600183` | 0.180000 |
| `600584` | 0.180000 |
| `688396` | 0.180000 |

## After Sync: Aggressive Score Result

| stock_id | weight |
|---|---:|
| `000792` | 0.200000 |
| `600233` | 0.200000 |
| `601669` | 0.200000 |
| `600930` | 0.200000 |
| `002463` | 0.200000 |

## Single-Slice Score Recheck

- aggressive output score: `0.077484`
- case zip current score: `0.025179`
- diff vs case current: `+0.052304`
- case zip best score: `0.037838`
- diff vs case best: `+0.039646`

## Validation Steps

| step | status | returncode | command |
|---|---|---:|---|
| `result_validator` | `PASS` | 0 | `D:\456\python.exe app/code/src/result_validator.py --result_path app/output/result.csv` |
| `pre_submit_check` | `PASS` | 0 | `D:\456\python.exe app/code/src/pre_submit_check.py --root_dir . --result_path app/output/result.csv` |
| `single_slice_score_recheck` | `PASS` | 0 | `D:\456\python.exe app/code/src/compare_with_case_score.py --our_result_path app/output/result.csv --output_dir D:\Desktop\股票分析预测代码\app\model\aggressive_score_submission_candidate\case_score_recheck` |

## Manual Decision Note

- 若目标是比赛冲分，可以继续使用当前 `app/output/result.csv`。
- 若目标是稳定策略，应执行上面的 rollback command 回到 HV rerank 默认结果。
- 本脚本没有修改 `app/model/default_submission_config.json`、`best_config.json` 或 `model_meta.json`。
