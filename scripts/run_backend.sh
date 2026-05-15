#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export EVOTENSOR_BASE_DIR="${EVOTENSOR_BASE_DIR:-$ROOT_DIR}"
export EVOTENSOR_VOLUMES_DIR="${EVOTENSOR_VOLUMES_DIR:-$ROOT_DIR/volumes}"
export EVOTENSOR_AUTH_MODE="${EVOTENSOR_AUTH_MODE:-dev}"

if [ -d "/opt/homebrew/opt/expat/lib" ]; then
  export DYLD_LIBRARY_PATH="/opt/homebrew/opt/expat/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi

mkdir -p "$EVOTENSOR_VOLUMES_DIR/projects" "$EVOTENSOR_VOLUMES_DIR/projects_by_user"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
fi

"$PYTHON_BIN" -c 'import sys; sys.exit("Python 3.11, 3.12, or 3.13 is required") if sys.version_info < (3, 11) or sys.version_info >= (3, 14) else None'

if [ "${EVOTENSOR_RELOAD:-0}" = "1" ]; then
  exec "$PYTHON_BIN" -m uvicorn backend.main:app --host 127.0.0.1 --port "${PORT:-8000}" --reload
fi

exec "$PYTHON_BIN" -m uvicorn backend.main:app --host 127.0.0.1 --port "${PORT:-8000}"
