#!/bin/bash
# Fine-tuning NeuroSTORM on ABIDE - AGGRESSIVE optimization for 8x A100-80GB
# Usage: bash scripts/abide_downstream/train_8gpu_aggressive.sh

# =============================================================================
# AGGRESSIVE CONFIGURATION FOR 8x A100-80GB
# =============================================================================

# Large batch size for A100-80GB
batch_size_per_gpu=16

# Calculate effective batch size
effective_batch_size=$((8 * batch_size_per_gpu))  # 128

# Learning rate scaling with warmup
base_lr=5e-5
base_batch_size=2
scaled_lr=$(echo "$base_lr * $effective_batch_size / $base_batch_size" | bc -l)

# Extended training for better convergence
max_epochs=50

# Maximum data loading workers
num_workers=16

echo "================================================================================"
echo "AGGRESSIVE Training Configuration for 8x A100-80GB"
echo "================================================================================"
echo "Batch size per GPU:        $batch_size_per_gpu"
echo "Effective batch size:      $effective_batch_size (8 GPUs × $batch_size_per_gpu)"
echo "Scaled learning rate:      $scaled_lr"
echo "Max epochs:                $max_epochs"
echo "Mixed precision:           16-bit (AMP)"
echo "Gradient clipping:         Enabled (0.5)"
echo "LR scheduler:              Cosine with warmup"
echo "================================================================================"

# =============================================================================
# GPU CONFIGURATION - OPTIMIZED FOR NVLINK/INFINIBAND
# =============================================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Optimized NCCL settings for A100 with NVLink
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0  # Adjust based on your network interface
export NCCL_DEBUG=WARN

# PyTorch optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

project_name="abide_ft_neurostorm_age_regression_aggressive"

# =============================================================================
# LAUNCH TRAINING WITH GRADIENT CLIPPING AND LR SCHEDULING
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
  --cycle 0.3 \
  --gamma 0.9 \
  --grad_clip \
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
# NOTES FOR AGGRESSIVE CONFIGURATION
# =============================================================================
# This configuration is designed to maximize performance on 8x A100-80GB:
#
# Key Features:
# 1. Very Large Batch Size (128 total)
#    - Better gradient estimates
#    - More stable training
#    - Requires careful LR scaling
#
# 2. Extended Training (50 epochs)
#    - Ensures full convergence
#    - Better final performance
#
# 3. Gradient Clipping (--grad_clip)
#    - Prevents gradient explosion
#    - Critical for large batch training
#
# 4. LR Schedule with Decay
#    - Cosine annealing with gamma=0.9
#    - Better final convergence
#
# 5. Mixed Precision (FP16)
#    - Essential for large batch sizes
#    - 2x speedup on A100
#
# Expected Performance Improvements vs. Baseline (2 GPU, batch=2, 10 epochs):
# - Training time: ~40% faster per epoch × 5x more epochs = 3x total time
# - Model quality: Significantly better (larger batch, more epochs, better LR)
# - R² score: Expected improvement from 0.02 to 0.3-0.6 (if data quality is good)
#
# If OOM Errors Occur:
# 1. Reduce batch_size_per_gpu to 12 or 8
# 2. Reduce num_workers to 8
# 3. Check if precision 16 is working (should reduce memory by ~50%)
#
# If Training is Unstable:
# 1. Reduce learning rate by 0.5x: scaled_lr × 0.5
# 2. Increase gradient clip value to 1.0
# 3. Add --augment_during_training for regularization
# =============================================================================
