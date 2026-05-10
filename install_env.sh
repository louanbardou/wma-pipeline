#!/bin/bash
# install_env.sh — Create self-contained venv for WMA pipeline
# Usage: bash install_env.sh [venv_path]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${1:-$SCRIPT_DIR/.venv}"

echo "Creating venv at $VENV ..."
python3 -m venv "$VENV"
source "$VENV/bin/activate"

pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "Done. Activate with:"
echo "  source $SCRIPT_DIR/activate_env.sh"
