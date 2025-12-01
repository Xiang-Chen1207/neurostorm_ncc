#!/usr/bin/env python3
"""
Calculate NeuroSTORM model parameters theoretically without loading PyTorch
Based on the architecture defined in models/neurostorm.py
"""

import math

def calculate_patch_embedding_params(in_chans, patch_size, embed_dim):
    """Calculate parameters for patch embedding layer"""
    # Linear layer: in_features = in_chans * patch_size[0] * patch_size[1] * patch_size[2] * patch_size[3]
    in_features = in_chans * patch_size[0] * patch_size[1] * patch_size[2] * patch_size[3]
    # Linear(in_features, embed_dim) -> weight: in_features * embed_dim, bias: embed_dim
    params = in_features * embed_dim + embed_dim
    return params

def calculate_positional_embedding_params(dim, patch_dim):
    """Calculate parameters for positional embedding"""
    d, h, w, t = patch_dim
    # pos_embed: (1, dim, d, h, w, 1)
    pos_params = dim * d * h * w * 1
    # time_embed: (1, dim, 1, 1, 1, t)
    time_params = dim * 1 * 1 * 1 * t
    return pos_params + time_params

def calculate_mamba_params(dim, d_state=16, d_conv=4, expand=2):
    """Calculate parameters for Mamba layer"""
    # Based on mamba-ssm implementation
    # Simplified calculation - Mamba has several components:
    # 1. Input projection: dim -> expand * dim
    # 2. Conv1d: expand * dim with kernel size d_conv
    # 3. x_proj: expand * dim -> d_state + d_state + expand * dim
    # 4. dt_proj: expand * dim -> expand * dim
    # 5. A_log, D parameters
    # 6. Output projection: expand * dim -> dim

    d_inner = expand * dim

    # in_proj
    in_proj = dim * d_inner + d_inner

    # conv1d
    conv = d_inner * d_conv

    # x_proj
    dt_rank = math.ceil(dim / 16)
    x_proj = d_inner * (dt_rank + d_state * 2)

    # dt_proj
    dt_proj = dt_rank * d_inner + d_inner

    # A_log (d_inner, d_state) and D (d_inner)
    A_D = d_inner * d_state + d_inner

    # out_proj
    out_proj = d_inner * dim + dim

    # norm (if exists)
    norm = dim

    total = in_proj + conv + x_proj + dt_proj + A_D + out_proj + norm

    return total

def calculate_window_attention_params(dim, num_heads, window_size):
    """Calculate parameters for WindowAttention layer"""
    # qkv: Linear(dim, dim * 3, bias=qkv_bias)
    qkv_params = dim * (dim * 3) + (dim * 3)  # assuming qkv_bias=True

    # proj: Linear(dim, dim)
    proj_params = dim * dim + dim

    total = qkv_params + proj_params
    return total

def calculate_mlp_params(dim, mlp_ratio=4.0):
    """Calculate parameters for MLP layer"""
    mlp_hidden_dim = int(dim * mlp_ratio)
    # Linear(dim, mlp_hidden_dim) + Linear(mlp_hidden_dim, dim)
    fc1_params = dim * mlp_hidden_dim + mlp_hidden_dim
    fc2_params = mlp_hidden_dim * dim + dim
    return fc1_params + fc2_params

def calculate_layer_norm_params(dim):
    """Calculate parameters for LayerNorm"""
    # weight and bias
    return 2 * dim

def calculate_swin_block_params(dim, num_heads, window_size, mlp_ratio=4.0):
    """Calculate parameters for one SwinTransformerBlock4D"""
    # norm1
    norm1 = calculate_layer_norm_params(dim)

    # mamba (instead of window attention in original)
    mamba = calculate_mamba_params(dim)

    # norm2
    norm2 = calculate_layer_norm_params(dim)

    # mlp
    mlp = calculate_mlp_params(dim, mlp_ratio)

    total = norm1 + mamba + norm2 + mlp
    return total

def calculate_patch_merging_params(dim, c_multiplier=2):
    """Calculate parameters for PatchMergingV2"""
    # reduction: Linear(8 * dim, c_multiplier * dim, bias=False)
    reduction = (8 * dim) * (c_multiplier * dim)

    # norm: LayerNorm(8 * dim)
    norm = calculate_layer_norm_params(8 * dim)

    return reduction + norm

def calculate_basic_layer_params(dim, depth, num_heads, window_size, c_multiplier, downsample=True, mlp_ratio=4.0):
    """Calculate parameters for one BasicLayer (stage)"""
    # All blocks in this layer
    blocks_params = 0
    for i in range(depth):
        blocks_params += calculate_swin_block_params(dim, num_heads, window_size, mlp_ratio)

    # Downsampling layer (if exists)
    downsample_params = 0
    if downsample:
        downsample_params = calculate_patch_merging_params(dim, c_multiplier)

    return blocks_params + downsample_params

def main():
    print("=" * 80)
    print("NeuroSTORM Model Parameter Calculation")
    print("=" * 80)

    # Default configuration
    img_size = (96, 96, 96, 20)
    in_chans = 1
    embed_dim = 24
    window_size = [4, 4, 4, 4]
    first_window_size = [2, 2, 2, 2]
    patch_size = [6, 6, 6, 1]
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    c_multiplier = 2
    mlp_ratio = 4.0

    print("\nConfiguration:")
    print("-" * 80)
    print(f"  Image size:            {img_size}")
    print(f"  Input channels:        {in_chans}")
    print(f"  Embedding dimension:   {embed_dim}")
    print(f"  Patch size:            {patch_size}")
    print(f"  Depths:                {depths}")
    print(f"  Number of heads:       {num_heads}")
    print(f"  Channel multiplier:    {c_multiplier}")
    print(f"  MLP ratio:             {mlp_ratio}")

    # Calculate patch dimensions
    patch_dim = [
        img_size[0] // patch_size[0],
        img_size[1] // patch_size[1],
        img_size[2] // patch_size[2],
        img_size[3] // patch_size[3]
    ]

    print(f"\n  Patch dimensions:      {patch_dim}")
    print(f"  Number of stages:      {len(depths)}")

    print("\n" + "=" * 80)
    print("Parameter Breakdown by Component:")
    print("=" * 80)

    total_params = 0

    # 1. Patch Embedding
    patch_embed_params = calculate_patch_embedding_params(in_chans, patch_size, embed_dim)
    total_params += patch_embed_params
    print(f"\n1. Patch Embedding:      {patch_embed_params:,} parameters")

    # 2. Positional Embeddings (one for each stage)
    pos_embed_params = 0
    current_patch_dim = patch_dim.copy()
    current_dim = embed_dim

    for i in range(len(depths)):
        stage_pos_params = calculate_positional_embedding_params(current_dim, current_patch_dim)
        pos_embed_params += stage_pos_params
        print(f"   Stage {i} pos embed:   {stage_pos_params:,} (dim={current_dim}, patch_dim={current_patch_dim})")

        # Update dimensions for next stage
        if i < len(depths) - 1:
            current_dim *= c_multiplier
            current_patch_dim = [current_patch_dim[0]//2, current_patch_dim[1]//2,
                                current_patch_dim[2]//2, current_patch_dim[3]]

    total_params += pos_embed_params
    print(f"\n2. Positional Embeddings: {pos_embed_params:,} parameters")

    # 3. Transformer Stages
    print(f"\n3. Transformer Stages:")
    stages_params = 0
    current_dim = embed_dim

    for i in range(len(depths)):
        # Last layer doesn't have downsampling
        has_downsample = (i < len(depths) - 1)

        # Use first_window_size for first stage, window_size for others
        ws = first_window_size if i == 0 else window_size

        stage_params = calculate_basic_layer_params(
            current_dim, depths[i], num_heads[i], ws,
            c_multiplier, has_downsample, mlp_ratio
        )
        stages_params += stage_params

        print(f"   Stage {i} (dim={current_dim}, depth={depths[i]}): {stage_params:,} parameters")

        # Update dim for next stage
        if i < len(depths) - 1:
            current_dim *= c_multiplier

    total_params += stages_params
    print(f"\n   Total stages:         {stages_params:,} parameters")

    # Summary
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"\nTotal parameters:        {total_params:,}")
    print(f"\nModel size estimates:")
    print(f"  FP32 (4 bytes/param):  {total_params * 4 / (1024**2):.2f} MB")
    print(f"  FP16 (2 bytes/param):  {total_params * 2 / (1024**2):.2f} MB")
    print(f"  INT8 (1 byte/param):   {total_params / (1024**2):.2f} MB")

    print("\n" + "=" * 80)
    print("\nNote: This is a theoretical calculation based on the architecture.")
    print("Actual parameter count may vary slightly due to implementation details.")
    print("=" * 80)

if __name__ == "__main__":
    main()
