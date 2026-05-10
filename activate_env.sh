#!/bin/bash
# activate_env.sh — Self-contained environment activation for WMA pipeline
# Source this at the top of every session / SLURM script:
#   source /path/to/WMA/activate_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${WMA_VENV:-$SCRIPT_DIR/.venv}"

# CUDA (HPC only, silently skip if unavailable)
module load cuda/12.5 2>/dev/null || true

# Activate venv
source "$VENV/bin/activate"

# Data paths — override with env vars if needed
export WMA_DIR="$SCRIPT_DIR"
export ABCD_IMAGING="${ABCD_IMAGING:-/mnt/scratch/user/$USER/abcd_leuko}"
export WMA_RUNS="${WMA_RUNS:-/mnt/scratch/user/$USER/wma_runs}"
export WMA_CACHE="${WMA_CACHE:-/mnt/scratch/user/$USER/wma_cache}"
export WMA_LOGS="${WMA_LOGS:-/mnt/scratch/user/$USER/wma_logs}"

echo "WMA environment activated"
echo "  Python : $(python --version 2>&1 | cut -d' ' -f2)"
echo "  WMA_DIR: $WMA_DIR"
echo "  DATA   : $ABCD_IMAGING"
