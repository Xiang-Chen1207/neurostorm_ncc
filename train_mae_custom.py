#!/usr/bin/env python3
"""
MAE Pretraining Script for Custom fMRI Data

This script provides MAE (Masked Autoencoder) pretraining for the NeuroSTORM model
using custom fMRI data specified in data.txt.

Usage:
    python train_mae_custom.py --data_txt_path data.txt --batch_size 8 --mask_ratio 0.5

For data with different dimensions (e.g., 96x96x96x200), adjust:
    --img_size 96 96 96 50 --sequence_length 50

Features:
    - Loads fMRI data from paths listed in data.txt
    - Supports .npz, .npy, .pt, and .nii.gz formats
    - Automatic data resizing to match img_size
    - Masked autoencoder reconstruction task
    - Window-based or random masking strategies
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='MAE Pretraining for fMRI Foundation Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data parameters
    parser.add_argument('--data_txt_path', type=str, default='data.txt',
                        help='Path to text file containing data file paths')
    parser.add_argument('--image_path', type=str, default='.',
                        help='Root directory for relative paths in data.txt')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=8,
                        help='Batch size for validation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--max_epochs', type=int, default=30,
                        help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    
    # MAE parameters
    parser.add_argument('--mask_ratio', type=float, default=0.5,
                        help='Ratio of masked tokens (0.0 to 1.0)')
    parser.add_argument('--spatial_mask', type=str, default='window',
                        choices=['window', 'random'],
                        help='Spatial masking strategy')
    parser.add_argument('--time_mask', type=str, default='random',
                        choices=['random', 'tube'],
                        help='Temporal masking strategy')
    
    # Model parameters  
    parser.add_argument('--sequence_length', type=int, default=40,
                        help='Number of time points to use')
    parser.add_argument('--img_size', type=int, nargs=4, default=[96, 96, 96, 40],
                        help='Image size (H W D T)')
    parser.add_argument('--embed_dim', type=int, default=36,
                        help='Embedding dimension')
    parser.add_argument('--depth', type=int, nargs='+', default=[2, 2, 6, 2],
                        help='Depth of each layer')
    parser.add_argument('--window_size', type=int, nargs=4, default=[4, 4, 4, 4],
                        help='Window size for attention')
    parser.add_argument('--first_window_size', type=int, nargs=4, default=[4, 4, 4, 4],
                        help='First layer window size')
    
    # Logging
    parser.add_argument('--project_name', type=str, default='custom_pt_neurostorm_mae',
                        help='Project name for logging')
    parser.add_argument('--loggername', type=str, default='tensorboard',
                        choices=['tensorboard', 'neptune'],
                        help='Logger to use')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--auto_resume', action='store_true',
                        help='Auto resume from last checkpoint')
    parser.add_argument('--gpus', type=str, default='0',
                        help='GPU IDs to use (comma-separated)')
    
    args = parser.parse_args()
    
    # Check if data.txt exists
    if not os.path.exists(args.data_txt_path):
        print(f"Error: {args.data_txt_path} not found!")
        print("Please create data.txt with one file path per line")
        sys.exit(1)
    
    # Count data files
    with open(args.data_txt_path, 'r') as f:
        data_files = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    if len(data_files) == 0:
        print(f"Error: No data files found in {args.data_txt_path}")
        print("Please add file paths to data.txt (one per line)")
        sys.exit(1)
    
    print(f"Found {len(data_files)} data files in {args.data_txt_path}")
    print(f"Starting MAE pretraining with:")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Mask ratio: {args.mask_ratio}")
    print(f"  - Sequence length: {args.sequence_length}")
    print(f"  - Image size: {args.img_size}")
    print(f"  - Project name: {args.project_name}")
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    os.environ['NCCL_P2P_DISABLE'] = '1'
    
    # Build command
    cmd_args = [
        'python', 'main.py',
        '--accelerator', 'gpu',
        '--max_epochs', str(args.max_epochs),
        '--num_nodes', '1',
        '--strategy', 'ddp',
        '--loggername', args.loggername,
        '--dataset_name', 'CustomMAE',
        '--image_path', args.image_path,
        '--batch_size', str(args.batch_size),
        '--eval_batch_size', str(args.eval_batch_size),
        '--num_workers', str(args.num_workers),
        '--project_name', args.project_name,
        '--c_multiplier', '2',
        '--last_layer_full_MSA', 'True',
        '--downstream_task_type', 'classification',
        '--pretraining',
        '--use_mae',
        '--spatial_mask', args.spatial_mask,
        '--time_mask', args.time_mask,
        '--mask_ratio', str(args.mask_ratio),
        '--dataset_split_num', '1',
        '--seed', str(args.seed),
        '--learning_rate', str(args.learning_rate),
        '--model', 'neurostorm',
        '--depth', *[str(d) for d in args.depth],
        '--embed_dim', str(args.embed_dim),
        '--sequence_length', str(args.sequence_length),
        '--img_size', *[str(s) for s in args.img_size],
        '--first_window_size', *[str(s) for s in args.first_window_size],
        '--window_size', *[str(s) for s in args.window_size],
    ]
    
    if args.auto_resume:
        cmd_args.append('--auto_resume')
    
    # Execute
    print(f"\nRunning command:")
    print(' '.join(cmd_args))
    print()
    
    os.execvp(cmd_args[0], cmd_args)

if __name__ == '__main__':
    main()
