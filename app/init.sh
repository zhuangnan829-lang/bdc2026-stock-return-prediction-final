#!/bin/bash
set -euo pipefail

echo "[init] starting environment checks..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "/app/code" ]; then
  APP_ROOT="/app"
else
  APP_ROOT="${SCRIPT_DIR}"
fi
CODE_ROOT="${APP_ROOT}/code"
SRC_ROOT="${CODE_ROOT}/src"
MODEL_DIR="${APP_ROOT}/model"
DATA_DIR="${APP_ROOT}/data"
OUTPUT_DIR="${APP_ROOT}/output"
TEMP_DIR="${APP_ROOT}/temp"
PYTHON_BIN="${PYTHON_BIN:-python}"

for dir in "${CODE_ROOT}" "${SRC_ROOT}" "${MODEL_DIR}" "${DATA_DIR}" "${OUTPUT_DIR}" "${TEMP_DIR}"; do
  if [ ! -d "${dir}" ]; then
    echo "[init] creating missing directory: ${dir}"
    mkdir -p "${dir}"
  fi
done

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[init] error: ${PYTHON_BIN} is not available" >&2
  exit 1
fi

echo "[init] app root: ${APP_ROOT}"
echo "[init] python version: $("${PYTHON_BIN}" --version 2>&1)"
echo "[init] code root: ${CODE_ROOT}"
echo "[init] data dir: ${DATA_DIR}"
echo "[init] output dir: ${OUTPUT_DIR}"
echo "[init] temp dir: ${TEMP_DIR}"

if [ ! -f "${DATA_DIR}/train.csv" ]; then
  echo "[init] warning: ${DATA_DIR}/train.csv not found yet"
fi

if [ ! -f "${DATA_DIR}/test.csv" ]; then
  echo "[init] warning: ${DATA_DIR}/test.csv not found yet"
fi

echo "[init] environment checks completed."
