#!/bin/bash
#SBATCH --job-name=wma_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/scratch/user/lbardou/wma_logs/wma_train_%A_%a.out
#SBATCH --error=/mnt/scratch/user/lbardou/wma_logs/wma_train_%A_%a.err
#SBATCH --array=0-4

set -euo pipefail

SCRIPT_DIR="/mnt/fac/CX500007_DS1/bardou/wma-pipeline"
source "$SCRIPT_DIR/activate_env.sh"

mkdir -p "$WMA_RUNS" "$WMA_CACHE" "$WMA_LOGS"

FOLD=${SLURM_ARRAY_TASK_ID:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$WMA_RUNS/fold${FOLD}_${TIMESTAMP}"
MANIFEST="$SCRIPT_DIR/data/manifest.csv"

mkdir -p "$RUN_DIR"

echo "============================================"
echo "WMA Training — Fold $FOLD"
echo "Run dir: $RUN_DIR"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

# manifest.csv is pre-built in data/
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest.csv not found at $MANIFEST" >&2
    exit 1
fi

# Train
python "$SCRIPT_DIR/wma_pipeline.py" train \
    --manifest "$MANIFEST" \
    --out_dir "$RUN_DIR" \
    --cache_dir "$WMA_CACHE" \
    --backbone resnet \
    --epochs 40 \
    --warmup_epochs 15 \
    --batch_size 4 \
    --effective_batch_size 16 \
    --lr 3e-4 \
    --dropout 0.4 \
    --fold "$FOLD" \
    --seed 42 \
    --freeze_epochs 5 \
    --weight_decay 5e-4 \
    --aploss_gamma 0.9 \
    --epoch_decay 1e-3 \
    --use_ema \
    --use_bf16 \
    --patience 10

# Evaluate with TTA
python "$SCRIPT_DIR/wma_pipeline.py" eval \
    --manifest "$MANIFEST" \
    --checkpoint "$RUN_DIR/best_model.pt" \
    --fold "$FOLD" \
    --use_tta \
    --use_bf16

echo "============================================"
echo "Training complete: $(date)"
echo "Best model: $RUN_DIR/best_model.pt"
echo "============================================"
