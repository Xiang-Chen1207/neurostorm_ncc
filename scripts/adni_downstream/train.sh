#!/bin/bash
# Fine-tuning NeuroSTORM on ADNI dataset for AD vs CN classification
# Usage: bash scripts/adni_downstream/ft_neurostorm_adni_ad_classification.sh [batch_size]

# Set default batch_size
batch_size="2"

# Override with the arguments if provided
if [ ! -z "$1" ]; then
  batch_size=$1
fi

# Set CUDA devices (modify as needed)
export CUDA_VISIBLE_DEVICES=4,3
export NCCL_P2P_DISABLE=1

# Construct project_name
project_name="adni_ft_neurostorm_ad_classification"

python /home/chenx/code/NeuroSTORM-main/main.py \
  --accelerator gpu \
  --max_epochs 10 \
  --num_nodes 1 \
  --strategy ddp \
  --loggername tensorboard \
  --clf_head_version v1 \
  --dataset_name ADNI \
  --image_path /home/chenx/code/NeuroSTORM-main/data \
  --batch_size "$batch_size" \
  --num_workers 8 \
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
  --load_model_path /home/chenx/code/NeuroSTORM-main/pt_fmrifound_mae_ratio0.5.ckpt \
  --num_sanity_val_steps 0 \
  --freeze_feature_extractor

# Notes:
# - The model will load .nii.gz files directly from the paths specified in the txt files
# - Each .nii.gz file is split into 20-frame segments continuously (0-19, 20-39, 40-59, ...)
# - Remaining frames that don't fit into a complete 20-frame segment are discarded
# - Labels (AD=1, CN=0) are extracted automatically from file paths containing 'ad' or 'cn'
# - Adjust --batch_size based on your GPU memory (default: 4)
# - Adjust CUDA_VISIBLE_DEVICES based on available GPUs
# - Pre-trained model path: ./output/neurostorm/pt_neurostorm_mae_ratio0.5.ckpt