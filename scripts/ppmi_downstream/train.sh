#!/bin/bash
# Fine-tuning NeuroSTORM on PPMI dataset for 3-class classification (PD, Control, Other)
# Usage: bash scripts/ppmi_downstream/train.sh [batch_size]

# Set default batch_size
batch_size="2"

# Override with the arguments if provided
if [ ! -z "$1" ]; then
  batch_size=$1
fi

# Set CUDA devices (modify as needed)
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1

# Construct project_name
project_name="ppmi_ft_neurostorm_3class_epoch20"

python /home/user/neurostorm_ncc/main.py \
  --accelerator gpu \
  --max_epochs 20 \
  --num_nodes 1 \
  --strategy ddp \
  --loggername tensorboard \
  --clf_head_version v1 \
  --dataset_name PPMI \
  --image_path /home/user/neurostorm_ncc/data/ppmi \
  --batch_size "$batch_size" \
  --num_workers 4 \
  --eval_batch_size "$batch_size" \
  --project_name "$project_name" \
  --c_multiplier 2 \
  --last_layer_full_MSA True \
  --downstream_task_id 3 \
  --downstream_task_type "classification" \
  --num_classes 3 \
  --task_name "group_classification" \
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
  --load_model_path /home/user/neurostorm_ncc/pt_fmrifound_mae_ratio0.5.ckpt \
  --num_sanity_val_steps 0

# Notes:
# - The model will load .npz files directly from the paths specified in the txt files
# - Only the first 20 frames are used from each .npz file
# - Only the FIRST .npz file per subject is used (when multiple segments exist)
# - Labels are extracted from ppmi.csv based on Subject ID and Group_idx column
# - Group_idx values: 1 (PD), 2 (Control), 3 (Other) are converted to 0-indexed: 0, 1, 2
# - Subject ID is extracted from filename (e.g., "sub-294308_ses-01_..._seg005.npz" -> "294308")
# - Adjust --batch_size based on your GPU memory (default: 2)
# - Adjust CUDA_VISIBLE_DEVICES based on available GPUs
# - Pre-trained model path: /home/user/neurostorm_ncc/pt_fmrifound_mae_ratio0.5.ckpt
# - Metrics used: ACC (Accuracy) and F1-weighted
# - Three-class classification: Class 0 (PD), Class 1 (Control), Class 2 (Other)
