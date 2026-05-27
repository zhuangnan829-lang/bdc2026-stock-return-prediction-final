#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-8501}"

echo "[run_demo.sh] starting Streamlit demo on port ${PORT}"
python -m streamlit run "${SCRIPT_DIR}/streamlit_app.py" --server.address 127.0.0.1 --server.port "${PORT}" --server.headless true --browser.gatherUsageStats false
