#!/usr/bin/env python3
"""
Script to calculate the size of the NeuroSTORM model
"""

import torch
from models.neurostorm import NeuroSTORM

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def main():
    print("=" * 80)
    print("NeuroSTORM Model Size Analysis")
    print("=" * 80)

    # Default configuration from lightning_model.py
    config = {
        'img_size': (96, 96, 96, 20),  # Default spatial + temporal size
        'in_chans': 1,
        'embed_dim': 24,
        'window_size': [4, 4, 4, 4],
        'first_window_size': [2, 2, 2, 2],
        'patch_size': [6, 6, 6, 1],
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'mlp_ratio': 4.0,
        'qkv_bias': True,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.0,
        'c_multiplier': 2,
        'last_layer_full_MSA': False,
        'num_classes': 2
    }

    print("\nModel Configuration:")
    print("-" * 80)
    for key, value in config.items():
        print(f"  {key:25s}: {value}")

    print("\n" + "=" * 80)
    print("Creating model...")
    print("=" * 80)

    # Create model
    model = NeuroSTORM(**config)

    # Count parameters
    total_params, trainable_params = count_parameters(model)

    # Get model size
    model_size_mb = get_model_size(model)

    print("\nModel Statistics:")
    print("-" * 80)
    print(f"  Total parameters:      {total_params:,}")
    print(f"  Trainable parameters:  {trainable_params:,}")
    print(f"  Non-trainable params:  {total_params - trainable_params:,}")
    print(f"\n  Model size:            {model_size_mb:.2f} MB")
    print(f"  Model size:            {model_size_mb/1024:.4f} GB")

    # Calculate size for different precisions
    print("\nModel size in different precisions:")
    print("-" * 80)
    print(f"  FP32 (float32):        {model_size_mb:.2f} MB")
    print(f"  FP16 (float16):        {model_size_mb/2:.2f} MB")
    print(f"  INT8:                  {model_size_mb/4:.2f} MB")

    # Layer-wise breakdown
    print("\nLayer-wise parameter breakdown:")
    print("-" * 80)
    layer_params = {}
    for name, param in model.named_parameters():
        layer_type = name.split('.')[0]
        if layer_type not in layer_params:
            layer_params[layer_type] = 0
        layer_params[layer_type] += param.numel()

    for layer_name, num_params in sorted(layer_params.items(), key=lambda x: x[1], reverse=True):
        percentage = (num_params / total_params) * 100
        print(f"  {layer_name:25s}: {num_params:15,} ({percentage:6.2f}%)")

    print("\n" + "=" * 80)

    # Calculate feature dimensions at each stage
    print("\nFeature dimensions at each stage:")
    print("-" * 80)
    embed_dim = config['embed_dim']
    c_mult = config['c_multiplier']
    for i, depth in enumerate(config['depths']):
        dim = embed_dim * (c_mult ** i) if i > 0 else embed_dim
        print(f"  Stage {i}: depth={depth}, dim={dim}, num_heads={config['num_heads'][i]}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
