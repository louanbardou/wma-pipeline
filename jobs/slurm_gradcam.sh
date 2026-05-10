#!/bin/bash
#SBATCH --job-name=wma_gradcam
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/wma_gradcam_%j.out
#SBATCH --error=logs/wma_gradcam_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$SCRIPT_DIR/activate_env.sh"

CHECKPOINT="${1:?Usage: sbatch slurm_gradcam.sh /path/to/best_model.pt [fold]}"
FOLD="${2:-0}"
OUT_DIR="$WMA_RUNS/gradcam_fold${FOLD}_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUT_DIR"

echo "GradCAM — Fold $FOLD | Checkpoint: $CHECKPOINT | Output: $OUT_DIR"

python "$SCRIPT_DIR/wma_gradcam.py" \
    --checkpoint "$CHECKPOINT" \
    --manifest "$SCRIPT_DIR/data/manifest.csv" \
    --fold "$FOLD" \
    --out_dir "$OUT_DIR"

echo "Done: $(date)"
