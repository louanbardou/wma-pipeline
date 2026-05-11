#!/bin/bash
#SBATCH --job-name=wma_swin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/scratch/user/lbardou/wma_logs/wma_swin_%A_%a.out
#SBATCH --error=/mnt/scratch/user/lbardou/wma_logs/wma_swin_%A_%a.err
##SBATCH --array=0-4

set -euo pipefail

SCRIPT_DIR="/mnt/fac/CX500007_DS1/bardou/wma-pipeline"
source "$SCRIPT_DIR/activate_env.sh"

mkdir -p "$WMA_RUNS" "$WMA_CACHE" "$WMA_LOGS"

FOLD=${SLURM_ARRAY_TASK_ID:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$WMA_RUNS/swin_fold${FOLD}_${TIMESTAMP}"
MANIFEST="$SCRIPT_DIR/data/manifest.csv"

mkdir -p "$RUN_DIR"

echo "============================================"
echo "WMA Swin Training — Fold $FOLD"
echo "Run dir: $RUN_DIR"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest.csv not found at $MANIFEST" >&2
    exit 1
fi

python "$SCRIPT_DIR/wma_pipeline_swin.py" train \
    --manifest "$MANIFEST" \
    --out_dir "$RUN_DIR" \
    --cache_dir "$WMA_CACHE" \
    --epochs 40 \
    --warmup_epochs 10 \
    --batch_size 2 \
    --effective_batch_size 16 \
    --lr 1e-4 \
    --dropout 0.4 \
    --fold "$FOLD" \
    --seed 42 \
    --freeze_epochs 10 \
    --weight_decay 5e-4 \
    --aploss_gamma 0.9 \
    --epoch_decay 1e-3 \
    --patience 10

echo "============================================"
echo "Training complete: $(date)"
echo "Best model: $RUN_DIR/best_model.pt"
echo "============================================"
