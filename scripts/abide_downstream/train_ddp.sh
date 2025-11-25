#!/bin/bash
# Fine-tuning NeuroSTORM on ABIDE dataset for age regression (DDP version with 2 GPUs)
# Usage: bash scripts/abide_downstream/train_ddp.sh [batch_size]

# Set default batch_size - increase for better GPU utilization with DDP
batch_size="4"

# Override with the arguments if provided
if [ ! -z "$1" ]; then
  batch_size=$1
fi

# Set CUDA devices (modify as needed)
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1

# Construct project_name
project_name="abide_ft_neurostorm_age_regression_ddp"

python /home/chenx/code/neurostorm_ncc/main.py \
  --accelerator gpu \
  --devices 2 \
  --max_epochs 10 \
  --num_nodes 1 \
  --strategy ddp \
  --loggername tensorboard \
  --clf_head_version v1 \
  --dataset_name ABIDE \
  --image_path /home/chenx/code/neurostorm_ncc/data/abide \
  --batch_size "$batch_size" \
  --num_workers 8 \
  --eval_batch_size "$batch_size" \
  --project_name "$project_name" \
  --c_multiplier 2 \
  --last_layer_full_MSA True \
  --downstream_task_id 1 \
  --downstream_task_type "regression" \
  --task_name "age" \
  --dataset_split_num 1 \
  --seed 1234 \
  --learning_rate 5e-5 \
  --model neurostorm \
  --depth 2 2 6 2 \
  --embed_dim 36 \
  --sequence_length 20 \
  --img_size 96 96 96 20 \
  --first_window_size 4 4 4 4 \
  --window_size 4 4 4 4 \
  --load_model_path /home/chenx/code/neurostorm_ncc/pt_fmrifound_mae_ratio0.5.ckpt \
  --num_sanity_val_steps 0

# Notes for DDP version:
# - Uses 2 GPUs for distributed training
# - Increased default batch_size to 4 for better GPU utilization
# - Validation set: 88 samples / 2 GPUs = 44 samples per GPU
# - Test set: 176 samples / 2 GPUs = 88 samples per GPU
# - With batch_size=4: validation needs 11 batches per GPU, test needs 22 batches
# - The empty tensor check in lightning_model.py will handle edge cases
