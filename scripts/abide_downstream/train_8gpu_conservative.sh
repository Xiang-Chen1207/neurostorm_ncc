#!/bin/bash
# CONSERVATIVE: 8-GPU training with moderate LR scaling
# For users who want safer defaults

batch_size_per_gpu="${1:-8}"
effective_batch_size=$((8 * batch_size_per_gpu))

# CONSERVATIVE: Only 4x scaling instead of 16x
# Original: 5e-5 for batch=4
# This: 2e-4 for batch=64 (4x instead of 16x)
base_lr=5e-5
conservative_factor=4  # Instead of full 16x scaling
scaled_lr=$(echo "$base_lr * $conservative_factor" | bc -l)

max_epochs=40  # Slightly more epochs to compensate
num_workers=16

echo "================================================================================"
echo "CONSERVATIVE Training Configuration for 8x A100"
echo "================================================================================"
echo "Batch size per GPU:        $batch_size_per_gpu"
echo "Effective batch size:      $effective_batch_size"
echo "Learning rate:             $scaled_lr (conservative 4x scaling)"
echo "Max epochs:                $max_epochs"
echo "================================================================================"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=0

project_name="abide_ft_neurostorm_age_regression_conservative"

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

# Notes:
# - Conservative LR scaling: 5e-5 → 2e-4 (4x instead of 16x)
# - More stable training, less risk of divergence
# - Slightly longer training (40 epochs) to compensate
# - Good for first-time 8-GPU training
