#!/usr/bin/env bash
# run.sh — launch the OTTER Debugger inside otter-env
# Usage: bash otter/run.sh   (from the AutoEncoders/ directory)
#    or: chmod +x otter/run.sh && ./otter/run.sh  (from otter/)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../otter-env"

if [ ! -f "$VENV/bin/python" ]; then
  echo "ERROR: otter-env not found at $VENV"
  echo "Run: python3 -m venv otter-env && otter-env/bin/pip install -r otter/requirements.txt flask"
  exit 1
fi

echo "Using Python: $VENV/bin/python"
"$VENV/bin/python" "$SCRIPT_DIR/app.py"
