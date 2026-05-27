#!/bin/bash
set -euo pipefail

echo "[app/data/run.sh] competition-compatible entrypoint started"
bash /app/run_submission.sh
