#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "/app/code" ]; then
  APP_ROOT="/app"
else
  APP_ROOT="${SCRIPT_DIR}"
fi
CODE_ROOT="${APP_ROOT}/code"
SRC_ROOT="${CODE_ROOT}/src"
MODEL_DIR="${APP_ROOT}/model"
TEMP_DIR="${APP_ROOT}/temp"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_BACKTEST="${RUN_BACKTEST:-1}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
RUN_RESULT_VALIDATION="${RUN_RESULT_VALIDATION:-1}"

echo "[run_research_pipeline] starting research pipeline..."
echo "[run_research_pipeline] step 1/4: train + walk-forward validation"
bash "${APP_ROOT}/train.sh"

echo "[run_research_pipeline] step 2/4: frozen inference snapshot"
bash "${APP_ROOT}/test.sh"

if [ "${RUN_RESULT_VALIDATION}" = "1" ]; then
  echo "[run_research_pipeline] validating result snapshot"
  "${PYTHON_BIN}" "${SRC_ROOT}/result_validator.py" --result_path "${APP_ROOT}/output/result.csv"
fi

if [ "${RUN_BACKTEST}" = "1" ]; then
  echo "[run_research_pipeline] step 3/4: local backtest"
  "${PYTHON_BIN}" "${SRC_ROOT}/backtest.py" \
    --prediction_path "${MODEL_DIR}/walk_forward_predictions.csv" \
    --feature_path "${TEMP_DIR}/train_features.csv" \
    --model_dir "${MODEL_DIR}" \
    --output_dir "${MODEL_DIR}" \
    --compare_profiles 1
else
  echo "[run_research_pipeline] step 3/4 skipped because RUN_BACKTEST=${RUN_BACKTEST}"
fi

if [ "${RUN_DIAGNOSTICS}" = "1" ]; then
  echo "[run_research_pipeline] step 4/4: fold diagnostics"
  "${PYTHON_BIN}" - <<'PY'
import sys
from pathlib import Path

app_root = Path("/app") if Path("/app/code").exists() else Path.cwd() / "app"
src_root = app_root / "code" / "src"
sys.path.insert(0, str(src_root))

from stability_diagnostics import (  # noqa: E402
    build_analysis_config,
    build_fold_diagnostics,
    load_prediction_artifact,
    write_fold_prediction_exports,
)

model_dir = app_root / "model"
temp_dir = app_root / "temp"
prediction_df = load_prediction_artifact(
    prediction_path=model_dir / "walk_forward_predictions.csv",
    feature_path=temp_dir / "train_features.csv",
)
write_fold_prediction_exports(prediction_df, model_dir)
config = build_analysis_config(profile_name="walk_forward_default")
fold_df, daily_df = build_fold_diagnostics(prediction_df, config)
fold_df.to_csv(model_dir / "fold_diagnostics.csv", index=False, encoding="utf-8-sig")
daily_df.to_csv(model_dir / "fold_daily_diagnostics.csv", index=False, encoding="utf-8-sig")
print("[run_research_pipeline] wrote fold diagnostics artifacts")
PY
else
  echo "[run_research_pipeline] step 4/4 skipped because RUN_DIAGNOSTICS=${RUN_DIAGNOSTICS}"
fi

echo "[run_research_pipeline] completed."
