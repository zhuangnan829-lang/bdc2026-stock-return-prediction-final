#!/bin/bash
set -euo pipefail

echo "[docker_rehearsal] starting full offline rehearsal..."

bash /app/run_submission.sh

python /app/code/src/result_validator.py --result_path /app/output/result.csv
python /app/code/src/pre_submit_check.py --root_dir / --result_path app/output/result.csv

echo "[docker_rehearsal] final result.csv"
cat /app/output/result.csv
echo "[docker_rehearsal] rehearsal passed"
