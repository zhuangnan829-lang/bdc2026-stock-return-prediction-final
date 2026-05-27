#!/bin/bash
set -euo pipefail

echo "[run_submission] starting formal submission entrypoint..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "/app/code" ]; then
  APP_ROOT="/app"
else
  APP_ROOT="${SCRIPT_DIR}"
fi
SRC_ROOT="${APP_ROOT}/code/src"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TRAIN="${RUN_TRAIN:-0}"

bash "${APP_ROOT}/init.sh"

echo "[run_submission] sync submission config..."
"${PYTHON_BIN}" "${SRC_ROOT}/sync_submission_config.py"

echo "[run_submission] compare config consistency..."
"${PYTHON_BIN}" "${SRC_ROOT}/compare_config_consistency.py"

if [ "${RUN_TRAIN}" = "1" ]; then
  echo "[run_submission] training enabled for this run"
  bash "${APP_ROOT}/train.sh"
else
  echo "[run_submission] using frozen submission artifacts because RUN_TRAIN=${RUN_TRAIN}"
fi

echo "[run_submission] running frozen inference"
bash "${APP_ROOT}/test.sh"

AGGRESSIVE_RESULT_PATH="${APP_ROOT}/model/aggressive_score_submission_candidate/result_aggressive_score.csv"
PACKAGE_VARIANT_PATH="${APP_ROOT}/model/package_variant.json"
if [ -f "${PACKAGE_VARIANT_PATH}" ] && [ -f "${AGGRESSIVE_RESULT_PATH}" ]; then
  if "${PYTHON_BIN}" - "${PACKAGE_VARIANT_PATH}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
sys.exit(0 if payload.get("variant") == "aggressive_score_submission" else 1)
PY
  then
    echo "[run_submission] aggressive score package detected; using frozen aggressive result"
    cp "${AGGRESSIVE_RESULT_PATH}" "${APP_ROOT}/output/result.csv"
  fi
fi

echo "[run_submission] running result validation"
"${PYTHON_BIN}" "${SRC_ROOT}/result_validator.py" --result_path "${APP_ROOT}/output/result.csv"

echo "[run_submission] final result.csv"
cat "${APP_ROOT}/output/result.csv"
echo "[run_submission] submission entrypoint completed"
