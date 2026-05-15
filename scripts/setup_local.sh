#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ -d "/opt/homebrew/opt/expat/lib" ]; then
  export DYLD_LIBRARY_PATH="/opt/homebrew/opt/expat/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi

"$PYTHON_BIN" -c 'import sys; sys.exit("Python 3.11, 3.12, or 3.13 is required") if sys.version_info < (3, 11) or sys.version_info >= (3, 14) else None'
"$PYTHON_BIN" -m venv .venv

"$ROOT_DIR/.venv/bin/python" -m pip install --upgrade pip
"$ROOT_DIR/.venv/bin/pip" install -r backend/requirements.txt

cd "$ROOT_DIR/frontend/evotensor-pro-ui"
npm install

echo "Local setup complete."
