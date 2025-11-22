#!/bin/bash
# Fine-tuning NeuroSTORM on ABIDE dataset for age regression
# Usage: bash scripts/abide_downstream/train.sh [batch_size]

# Set default batch_size
batch_size="2"

# Override with the arguments if provided
if [ ! -z "$1" ]; then
  batch_size=$1
fi

# Set CUDA devices (modify as needed)
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1

# Construct project_name
project_name="abide_ft_neurostorm_age_regression"

python /home/chenx/code/neurostorm_ncc/main.py \
  --accelerator gpu \
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

# Notes:
# - The model will load .npz files directly from the paths specified in the txt files
# - Only the first 20 frames from the first npz file of each subject are used
# - Labels (continuous age values) are extracted from AGE_AT_SCAN column in abide.csv
# - Subject ID is extracted from directory name (e.g., CMU_a_0050642_func_preproc -> 50642)
# - Adjust --batch_size based on your GPU memory (default: 2)
# - Adjust CUDA_VISIBLE_DEVICES based on available GPUs
# - Pre-trained model path: /home/user/neurostorm_ncc/pt_fmrifound_mae_ratio0.5.ckpt
# - Output will include predictions CSV with predicted and true age values
# - Metrics CSV will include Pearson correlation (R²) and MSE
# - This is an age regression task
