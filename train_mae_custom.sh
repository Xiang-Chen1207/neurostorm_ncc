#!/bin/bash
# MAE Pretraining Script for Custom fMRI Data
# This script trains the NeuroSTORM model using Masked Autoencoder (MAE) pretraining
# on custom fMRI data specified in data.txt

# Usage: bash train_mae_custom.sh [batch_size]
# Example: bash train_mae_custom.sh 8

# Default batch size
batch_size="8"

if [[ -n "$1" ]]; then
  batch_size=$1
fi

# GPU configuration
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1

# Project name for logging
project_name="custom_pt_neurostorm_mae0.5"

# Check if data.txt exists
if [[ ! -f "data.txt" ]]; then
    echo "Error: data.txt not found in current directory"
    echo "Please create data.txt with one file path per line"
    exit 1
fi

# Check if data.txt is readable
if [[ ! -r "data.txt" ]]; then
    echo "Error: data.txt is not readable"
    echo "Please check file permissions"
    exit 1
fi

# Count number of data files (safely handle errors)
num_files=$(grep -v "^#" data.txt 2>/dev/null | grep -v "^$" | wc -l || echo "0")
echo "Found $num_files data files in data.txt"

if [[ "$num_files" -eq 0 ]]; then
    echo "Error: No data files found in data.txt"
    echo "Please add file paths to data.txt (one per line)"
    exit 1
fi

echo "Starting MAE pretraining with batch_size=$batch_size"
echo "Project name: $project_name"

python main.py \
  --accelerator gpu \
  --max_epochs 30 \
  --num_nodes 1 \
  --strategy ddp \
  --loggername tensorboard \
  --dataset_name CustomMAE \
  --image_path . \
  --batch_size "$batch_size" \
  --eval_batch_size "$batch_size" \
  --num_workers 4 \
  --project_name "$project_name" \
  --c_multiplier 2 \
  --last_layer_full_MSA True \
  --downstream_task_type "classification" \
  --pretraining \
  --use_mae \
  --spatial_mask window \
  --time_mask random \
  --mask_ratio 0.5 \
  --dataset_split_num 1 \
  --seed 1 \
  --learning_rate 5e-5 \
  --model neurostorm \
  --depth 2 2 6 2 \
  --embed_dim 36 \
  --sequence_length 40 \
  --img_size 96 96 96 40 \
  --first_window_size 4 4 4 4 \
  --window_size 4 4 4 4 \
  --auto_resume

echo "Training complete!"
echo "Model checkpoints saved to: output/neurostorm/$project_name/"
