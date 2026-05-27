#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "/app/code" ]; then
  APP_ROOT="/app"
else
  APP_ROOT="${SCRIPT_DIR}"
fi
SRC_ROOT="${APP_ROOT}/code/src"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[freeze_submission] compare config consistency before freeze..."
"${PYTHON_BIN}" "${SRC_ROOT}/compare_config_consistency.py"

"${PYTHON_BIN}" "${SRC_ROOT}/cli.py" --app-root "${APP_ROOT}" freeze "$@"
