# MAE Pretraining for fMRI Foundation Model

This guide explains how to use the Masked Autoencoder (MAE) pretraining implementation for the NeuroSTORM fMRI foundation model with custom data.

## Overview

The MAE pretraining task learns representations by:
1. Randomly masking patches of the input fMRI volume
2. Encoding the unmasked patches
3. Reconstructing the original signal at masked locations
4. Computing reconstruction loss only on masked patches

## Data Format

### Expected Input Shape
- Each data file should contain 4D fMRI volumes: `(H, W, D, T)`
  - H, W, D: Spatial dimensions (height, width, depth)
  - T: Number of time points (temporal dimension)
- Example: `(96, 96, 96, 200)` - 96x96x96 volume with 200 time points

### Supported File Formats
- `.npz` - NumPy compressed array
- `.npy` - NumPy array
- `.pt` - PyTorch tensor
- `.nii.gz` / `.nii` - NIfTI format

## Quick Start

### 1. Prepare Your Data File List

Create a `data.txt` file with one data file path per line:

```bash
# data.txt example
/path/to/subject_001.npz
/path/to/subject_002.npy
/path/to/subject_003.pt
/path/to/subject_004.nii.gz
```

**Important Notes:**
- Use absolute paths or relative paths from the `--image_path` directory
- Lines starting with `#` are treated as comments
- Empty lines are ignored

### 2. Run Training

#### Option A: Using the Shell Script (Recommended)

```bash
# Basic usage with default settings
bash train_mae_custom.sh

# Specify batch size
bash train_mae_custom.sh 12

# Make script executable first (if needed)
chmod +x train_mae_custom.sh
```

#### Option B: Using the Python Script

```bash
# Basic usage
python train_mae_custom.py --data_txt_path data.txt --batch_size 8

# With custom parameters
python train_mae_custom.py \
  --data_txt_path data.txt \
  --batch_size 12 \
  --mask_ratio 0.5 \
  --max_epochs 50 \
  --learning_rate 1e-4 \
  --sequence_length 40 \
  --img_size 96 96 96 40
```

#### Option C: Direct Call to main.py

```bash
python main.py \
  --accelerator gpu \
  --max_epochs 30 \
  --dataset_name CustomMAE \
  --image_path . \
  --batch_size 8 \
  --pretraining \
  --use_mae \
  --spatial_mask window \
  --time_mask random \
  --mask_ratio 0.5 \
  --model neurostorm \
  --sequence_length 40 \
  --img_size 96 96 96 40 \
  --window_size 4 4 4 4 \
  --depth 2 2 6 2 \
  --embed_dim 36 \
  --learning_rate 5e-5 \
  --project_name custom_mae_pretraining
```

## Configuration Parameters

### Key MAE Parameters

- `--mask_ratio`: Proportion of tokens to mask (default: 0.5)
  - Range: 0.0 to 1.0
  - Higher values = more challenging reconstruction task
  - Recommended: 0.5 to 0.8

- `--spatial_mask`: Spatial masking strategy
  - `window`: Mask entire windows (recommended)
  - `random`: Mask random individual patches

- `--time_mask`: Temporal masking strategy
  - `random`: Random masking across time
  - `tube`: Mask temporal tubes

### Data Parameters

- `--sequence_length`: Number of time points to use per sample (default: 40)
  - Must be ≤ T dimension of your data
  - For data with shape (96,96,96,200), you can use up to 200

- `--img_size`: Target dimensions [H W D T] (default: [96, 96, 96, 40])
  - Data will be automatically resized if dimensions don't match
  - Last value should match `--sequence_length`

### Model Parameters

- `--embed_dim`: Embedding dimension (default: 36)
- `--depth`: Number of blocks per layer (default: [2, 2, 6, 2])
- `--window_size`: Attention window size (default: [4, 4, 4, 4])
- `--c_multiplier`: Channel multiplier between layers (default: 2)

### Training Parameters

- `--batch_size`: Training batch size (default: 8)
  - Adjust based on GPU memory
  - Larger batch = better training but more memory

- `--learning_rate`: Learning rate (default: 5e-5)
- `--max_epochs`: Maximum training epochs (default: 30)
- `--num_workers`: Data loading workers (default: 4)

## Adapting to Different Data Dimensions

If your data has shape (96, 96, 96, 200):

```bash
python train_mae_custom.py \
  --sequence_length 50 \
  --img_size 96 96 96 50 \
  --batch_size 8 \
  --mask_ratio 0.5
```

For larger sequences (e.g., using all 200 time points):

```bash
python train_mae_custom.py \
  --sequence_length 200 \
  --img_size 96 96 96 200 \
  --batch_size 4 \
  --mask_ratio 0.5
```

## Output and Checkpoints

Training outputs are saved to:
```
output/neurostorm/<project_name>/
├── checkpt-epoch=XX-valid_loss=Y.YY.ckpt  # Best checkpoint
├── last.ckpt                               # Latest checkpoint
└── tensorboard logs/                       # Training logs
```

View training progress:
```bash
tensorboard --logdir output/neurostorm/<project_name>/
```

## Custom Dataset Implementation Details

The `CustomMAE` dataset class:
- Loads file paths from `data.txt`
- Validates file existence and data shapes
- Creates multiple samples per file if it contains many time points
- Handles different file formats uniformly
- Supports automatic resizing to target dimensions

Key features:
- **Flexible loading**: Supports multiple file formats
- **Error handling**: Skips corrupted or invalid files
- **Progress tracking**: Shows loading progress
- **Memory efficient**: Loads data on-demand during training

## Troubleshooting

### Issue: "data.txt not found"
**Solution**: Create data.txt in the same directory as the training script, or specify full path with `--data_txt_path`

### Issue: "No data files found in data.txt"
**Solution**: Ensure data.txt contains at least one valid file path and is not all comments/empty lines

### Issue: "Out of memory"
**Solution**: Reduce `--batch_size` or `--sequence_length`

### Issue: "File not found" for data files
**Solution**: 
- Use absolute paths in data.txt, or
- Set `--image_path` to the directory containing your data files, and use relative paths in data.txt

### Issue: "Unexpected shape"
**Solution**: Verify your data files have shape (H, W, D, T) where T >= sequence_length

## Advanced Usage

### Resume Training from Checkpoint

```bash
python train_mae_custom.py --auto_resume
```

This will automatically find and resume from the last checkpoint.

### Multiple Mask Ratios

Train models with different mask ratios to find optimal setting:

```bash
# Light masking
python train_mae_custom.py --mask_ratio 0.3 --project_name mae_mask30

# Medium masking (recommended)
python train_mae_custom.py --mask_ratio 0.5 --project_name mae_mask50

# Heavy masking
python train_mae_custom.py --mask_ratio 0.8 --project_name mae_mask80
```

### Multi-GPU Training

```bash
# Use multiple GPUs
python train_mae_custom.py --gpus 0,1,2,3 --batch_size 32
```

## Reference

Based on the NeuroSTORM paper:
- Paper: "Towards a General-Purpose Foundation Model for fMRI Analysis"
- GitHub: https://github.com/CUHK-AIM-Group/NeuroSTORM

MAE architecture inspired by:
- He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022
