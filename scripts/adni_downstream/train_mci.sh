#!/bin/bash
# Fine-tuning NeuroSTORM on ADNI-MCI dataset for MCI vs CN classification
# Usage: bash scripts/adni_downstream/train_mci.sh [batch_size]

# Set default batch_size
batch_size="2"

# Override with the arguments if provided
if [ ! -z "$1" ]; then
  batch_size=$1
fi

# Set CUDA devices (modify as needed)
export CUDA_VISIBLE_DEVICES=4
export NCCL_P2P_DISABLE=1

# Construct project_name
project_name="adni_mci_ft_neurostorm_classification_epoch20"

python /home/chenx/code/neurostorm_ncc/main.py \
  --accelerator gpu \
  --max_epochs 20 \
  --num_nodes 1 \
  --strategy ddp \
  --loggername tensorboard \
  --clf_head_version v1 \
  --dataset_name ADNI_MCI \
  --image_path /home/chenx/code/neurostorm_ncc/data/adni_mci \
  --batch_size "$batch_size" \
  --num_workers 4 \
  --eval_batch_size "$batch_size" \
  --project_name "$project_name" \
  --c_multiplier 2 \
  --last_layer_full_MSA True \
  --downstream_task_id 3 \
  --downstream_task_type "classification" \
  --num_classes 2 \
  --task_name "diagnosis" \
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

# Notes:
# - The model will load .npz files directly from the paths specified in the txt files
# - Labels (MCI=1, CN=0) are extracted automatically from file paths containing 'mci' or 'cn'
# - Adjust --batch_size based on your GPU memory (default: 2)
# - Adjust CUDA_VISIBLE_DEVICES based on available GPUs
# - Pre-trained model path: /home/chenx/code/neurostorm_ncc/pt_fmrifound_mae_ratio0.5.ckpt
# - Metrics used: ACC (Accuracy) and F1-weighted
