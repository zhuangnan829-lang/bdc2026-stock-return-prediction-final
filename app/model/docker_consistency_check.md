# Docker Consistency Check

- generated_at_utc: `2026-05-27T01:32:33+00:00`
- status: `PASS`

## Result Summary

| item | local | docker | match |
|---|---:|---:|---|
| path | `app/output/result.csv` | `test/output/result.csv` |  |
| rows | 5 | 5 | PASS |
| weight_sum | 0.99999999999999994 | 0.99999999999999994 | PASS |
| result_md5 | `71401e2070a49f521bd40477fe0b5d16` | `71401e2070a49f521bd40477fe0b5d16` | PASS |

- local_result_md5: `71401e2070a49f521bd40477fe0b5d16`
- docker_result_md5: `71401e2070a49f521bd40477fe0b5d16`

## Row Comparison

| rank | local_stock_id | local_weight | docker_stock_id | docker_weight | match |
|---:|---|---:|---|---:|---|
| 1 | 000792 | 0.2 | 000792 | 0.2 | PASS |
| 2 | 600233 | 0.2 | 600233 | 0.2 | PASS |
| 3 | 601669 | 0.2 | 601669 | 0.2 | PASS |
| 4 | 600930 | 0.19999999999999998 | 600930 | 0.19999999999999998 | PASS |
| 5 | 002463 | 0.19999999999999996 | 002463 | 0.19999999999999996 | PASS |

## Checks

- stock_ids_match: `True`
- weights_match: `True`
- weight_sums_match: `True`
- md5_match: `True`
