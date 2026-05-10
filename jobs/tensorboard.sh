#!/bin/bash
# Usage: bash jobs/tensorboard.sh [logdir]
# Then: ssh -L 6006:localhost:6006 user@hpc-login → http://localhost:6006

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$SCRIPT_DIR/activate_env.sh"

LOGDIR="${1:-$WMA_RUNS}"
echo "TensorBoard: $LOGDIR → http://localhost:6006"
tensorboard --logdir "$LOGDIR" --port 6006 --bind_all
