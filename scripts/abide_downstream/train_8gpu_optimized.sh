#!/bin/bash
# Fine-tuning NeuroSTORM on ABIDE dataset for age regression - Optimized for 8x A100
# Usage: bash scripts/abide_downstream/train_8gpu_optimized.sh [batch_size_per_gpu]

# =============================================================================
# CONFIGURATION FOR 8x A100 GPUs
# =============================================================================

# Batch size per GPU (default: 8 for A100-40GB, 16 for A100-80GB)
batch_size_per_gpu="${1:-8}"

# Calculate effective batch size: 8 GPUs × batch_size_per_gpu
effective_batch_size=$((8 * batch_size_per_gpu))

# Learning rate scaling: Use linear scaling rule
# Base LR (for batch_size=2): 5e-5
# Effective batch size: 64 (8 GPUs × 8 per GPU)
# Scaled LR: 5e-5 × (64 / 2) = 1.6e-3
base_lr=5e-5
base_batch_size=2
scaled_lr=$(echo "$base_lr * $effective_batch_size / $base_batch_size" | bc -l)

# Training epochs (increased from 10 to 30 for better convergence)
max_epochs=30

# Number of data loading workers per GPU
num_workers=16

echo "================================================================================"
echo "Training Configuration for 8x A100 GPUs"
echo "================================================================================"
echo "Batch size per GPU:        $batch_size_per_gpu"
echo "Effective batch size:      $effective_batch_size (8 GPUs × $batch_size_per_gpu)"
echo "Scaled learning rate:      $scaled_lr (linear scaling from base $base_lr)"
echo "Max epochs:                $max_epochs"
echo "Num workers per GPU:       $num_workers"
echo "Mixed precision:           16-bit (automatic mixed precision)"
echo "================================================================================"

# =============================================================================
# GPU CONFIGURATION
# =============================================================================

# Use all 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NCCL settings for optimal multi-GPU performance
export NCCL_P2P_DISABLE=0  # Enable P2P if supported (better for A100)
export NCCL_IB_DISABLE=0   # Enable InfiniBand if available
export NCCL_DEBUG=INFO     # Show NCCL debug info (can set to WARN in production)

# Construct project_name
project_name="abide_ft_neurostorm_age_regression_8gpu_optimized"

# =============================================================================
# LAUNCH TRAINING
# =============================================================================

python /home/chenx/code/neurostorm_ncc/main.py \
  --accelerator gpu \
  --devices 8 \
  --max_epochs "$max_epochs" \
  --num_nodes 1 \
  --strategy ddp \
  --precision 16 \
  --loggername tensorboard \
  --clf_head_version v1 \
  --dataset_name ABIDE \
  --image_path /home/chenx/code/neurostorm_ncc/data/abide \
  --batch_size "$batch_size_per_gpu" \
  --num_workers "$num_workers" \
  --eval_batch_size "$batch_size_per_gpu" \
  --project_name "$project_name" \
  --c_multiplier 2 \
  --last_layer_full_MSA True \
  --downstream_task_id 1 \
  --downstream_task_type "regression" \
  --task_name "age" \
  --dataset_split_num 1 \
  --seed 1234 \
  --learning_rate "$scaled_lr" \
  --optimizer AdamW \
  --weight_decay 0.01 \
  --use_scheduler \
  --model neurostorm \
  --depth 2 2 6 2 \
  --embed_dim 36 \
  --sequence_length 20 \
  --img_size 96 96 96 20 \
  --first_window_size 4 4 4 4 \
  --window_size 4 4 4 4 \
  --load_model_path /home/chenx/code/neurostorm_ncc/pt_fmrifound_mae_ratio0.5.ckpt \
  --num_sanity_val_steps 0 \
  --eval_train_every 5

# =============================================================================
# NOTES
# =============================================================================
# Performance Optimizations Applied:
#
# 1. Multi-GPU Scaling (8x A100)
#    - Uses all 8 GPUs with DDP strategy
#    - Effective batch size: 64 (8 × 8)
#    - Expected speedup: ~7-8x over single GPU
#
# 2. Learning Rate Scaling
#    - Linear scaling rule: LR ∝ batch_size
#    - Base: 5e-5 @ batch_size=2
#    - Scaled: 1.6e-3 @ batch_size=64
#    - Prevents underfitting with large batch sizes
#
# 3. Mixed Precision Training (FP16)
#    - Uses automatic mixed precision (--precision 16)
#    - ~2x speedup + 50% memory reduction on A100
#    - Enables larger batch sizes
#
# 4. Increased Training Duration
#    - 30 epochs (vs. 10) for better convergence
#    - With 8 GPUs, still finishes faster than 10 epochs on 2 GPUs
#
# 5. Optimized Data Loading
#    - 16 workers per GPU (128 total)
#    - Minimizes data loading bottleneck
#
# 6. Cosine Annealing LR Schedule
#    - Enabled via --use_scheduler
#    - Improves final convergence quality
#
# Expected Training Time:
# - If 2 GPUs take 1 hour for 10 epochs
# - 8 GPUs with 30 epochs: ~1.1 hours (7-8x speedup)
#
# Memory Usage (per GPU):
# - FP32: ~25-30GB (may not fit batch_size=8 on A100-40GB)
# - FP16: ~12-15GB (easily fits batch_size=8, can try 12-16)
#
# Troubleshooting:
# - If OOM: reduce batch_size_per_gpu to 4 or 6
# - If training unstable: reduce learning rate by 0.5x
# - If convergence slow: increase max_epochs to 50
# =============================================================================
