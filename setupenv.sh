#!/usr/bin/env bash
# Create/refresh .venv for running simulator.py
# Usage: ./setupenv.sh            (then: source .venv/bin/activate)
#        PYTHON=python3.12 ./setupenv.sh   to pick a specific interpreter
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
VENV=".venv"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "error: '$PYTHON' not found. Install Python 3 or set PYTHON=..." >&2
    exit 1
fi

# Rebuild the venv if it is missing or its interpreter no longer works
if [ -d "$VENV" ] && ! "$VENV/bin/python" -c 'import sys' >/dev/null 2>&1; then
    echo "Existing $VENV is broken (interpreter missing) - recreating it."
    rm -rf "$VENV"
fi

if [ ! -d "$VENV" ]; then
    echo "Creating $VENV with $("$PYTHON" --version)..."
    "$PYTHON" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r requirements.txt

echo
echo "Done. Run the simulator with:"
echo "  source $VENV/bin/activate && python simulator.py"
echo "or:"
echo "  $VENV/bin/python simulator.py"
