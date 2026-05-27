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

"${PYTHON_BIN}" "${SRC_ROOT}/cli.py" --app-root "${APP_ROOT}" train "$@"
