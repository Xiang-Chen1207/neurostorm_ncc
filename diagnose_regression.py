#!/usr/bin/env python3
"""
Diagnostic script to check potential issues with ABIDE regression task.
Usage: python diagnose_regression.py
"""

import torch
import numpy as np
import pandas as pd
import os
from collections import OrderedDict

def check_model_output_head():
    """Check if output_head is properly initialized and has correct architecture."""
    print("="*80)
    print("1. Checking Regression Head Architecture")
    print("="*80)

    # Load the model
    from models.heads.reg_head import reg_head

    # Typical embedding size: embed_dim=36, c_multiplier=2, 4 stages -> 36*2^(4-1) = 36*8 = 288
    num_tokens = 36 * (2 ** 3)  # 288

    head = reg_head(version=1, num_tokens=num_tokens)
    print(f"✓ Regression head created with num_tokens={num_tokens}")
    print(f"  Architecture: {head}")

    # Check parameters
    total_params = sum(p.numel() for p in head.parameters())
    trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params}")
    print(f"  Trainable parameters: {trainable_params}")

    # Test forward pass
    batch_size = 2
    test_input = torch.randn(batch_size, num_tokens, 4, 4, 4, 20)
    output = head(test_input)
    print(f"  Test input shape: {test_input.shape}")
    print(f"  Test output shape: {output.shape}")
    print(f"  Expected output shape: ({batch_size}, 1)")

    if output.shape == (batch_size, 1):
        print("✓ Regression head output shape is correct!")
    else:
        print("✗ WARNING: Regression head output shape is incorrect!")
    print()

def check_data_loading():
    """Check if labels are loaded correctly."""
    print("="*80)
    print("2. Checking Data Loading and Labels")
    print("="*80)

    data_path = "/home/chenx/code/neurostorm_ncc/data/abide"
    csv_file = os.path.join(data_path, "abide.csv")

    if not os.path.exists(csv_file):
        print(f"✗ CSV file not found: {csv_file}")
        print("  Please check the path in train.sh script")
        return

    # Load CSV
    try:
        meta_data = pd.read_csv(csv_file)
        print(f"✓ CSV loaded successfully")
        print(f"  Columns: {list(meta_data.columns)}")
        print(f"  Number of rows: {len(meta_data)}")

        # Check age distribution
        if 'AGE_AT_SCAN' in meta_data.columns:
            ages = meta_data['AGE_AT_SCAN'].dropna()
            print(f"\n  Age statistics:")
            print(f"    Count: {len(ages)}")
            print(f"    Range: {ages.min():.2f} - {ages.max():.2f}")
            print(f"    Mean ± Std: {ages.mean():.2f} ± {ages.std():.2f}")
            print(f"    Median: {ages.median():.2f}")

            # Check if age distribution looks reasonable
            if ages.min() < 5 or ages.max() > 100:
                print(f"  ⚠ Warning: Age range looks unusual")
            else:
                print(f"  ✓ Age range looks reasonable")
        else:
            print(f"  ✗ 'AGE_AT_SCAN' column not found!")

        # Check subject IDs
        if 'SUB_ID' in meta_data.columns:
            print(f"\n  Subject ID statistics:")
            print(f"    Unique subjects: {meta_data['SUB_ID'].nunique()}")
            print(f"    Sample subject IDs: {list(meta_data['SUB_ID'].head())}")
        else:
            print(f"  ✗ 'SUB_ID' column not found!")

    except Exception as e:
        print(f"✗ Error loading CSV: {e}")

    print()

def check_standardization():
    """Check standardization behavior."""
    print("="*80)
    print("3. Checking Label Standardization")
    print("="*80)

    # Simulate age values from ABIDE (typically 5-65 years)
    ages = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50]).reshape(-1, 1)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    normalized_ages = scaler.fit_transform(ages)

    print(f"  Original ages: {ages.flatten()}")
    print(f"  Mean: {scaler.mean_[0]:.2f}, Std: {scaler.scale_[0]:.2f}")
    print(f"  Normalized ages: {normalized_ages.flatten()}")
    print(f"  Normalized range: [{normalized_ages.min():.2f}, {normalized_ages.max():.2f}]")

    # Check if a model predicting mean would give R²=0
    mean_prediction = np.zeros_like(normalized_ages)  # Predicting 0 (normalized mean)
    ss_res = np.sum((normalized_ages - mean_prediction) ** 2)
    ss_tot = np.sum((normalized_ages - np.mean(normalized_ages)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"\n  If model predicts normalized mean (0):")
    print(f"    R² = {r_squared:.4f}")
    print(f"    (R² should be 0 if model just predicts mean)")

    print()

def check_training_config():
    """Check training configuration."""
    print("="*80)
    print("4. Checking Training Configuration")
    print("="*80)

    # Read train.sh
    train_script = "/home/user/neurostorm_ncc/scripts/abide_downstream/train.sh"
    if os.path.exists(train_script):
        with open(train_script, 'r') as f:
            content = f.read()

        # Extract key parameters
        params = {
            'learning_rate': None,
            'max_epochs': None,
            'batch_size': None,
            'downstream_task_type': None,
            'freeze_feature_extractor': False
        }

        for line in content.split('\n'):
            if '--learning_rate' in line:
                try:
                    params['learning_rate'] = line.split('--learning_rate')[1].split()[0]
                except:
                    pass
            if '--max_epochs' in line:
                try:
                    params['max_epochs'] = line.split('--max_epochs')[1].split()[0]
                except:
                    pass
            if '--batch_size' in line:
                try:
                    params['batch_size'] = line.split('batch_size=')[1].split('"')[1]
                except:
                    pass
            if '--downstream_task_type' in line:
                try:
                    params['downstream_task_type'] = line.split('--downstream_task_type')[1].split('"')[1]
                except:
                    pass
            if '--freeze_feature_extractor' in line and not line.strip().startswith('#'):
                params['freeze_feature_extractor'] = True

        print("  Training parameters from train.sh:")
        for key, value in params.items():
            print(f"    {key}: {value}")

        # Check for potential issues
        print("\n  Potential issues:")
        issues_found = False

        if params['learning_rate']:
            lr = float(params['learning_rate'])
            if lr < 1e-6:
                print(f"    ⚠ Learning rate {lr} is very small")
                issues_found = True
            elif lr > 1e-3:
                print(f"    ⚠ Learning rate {lr} might be too large for fine-tuning")
                issues_found = True
            else:
                print(f"    ✓ Learning rate {lr} looks reasonable")

        if params['freeze_feature_extractor']:
            print(f"    ⚠ Feature extractor is FROZEN - only regression head will be trained")
            print(f"      This might limit model capacity")
            issues_found = True
        else:
            print(f"    ✓ Feature extractor is NOT frozen - full fine-tuning")

        if params['downstream_task_type'] != 'regression':
            print(f"    ✗ Task type is '{params['downstream_task_type']}', should be 'regression'!")
            issues_found = True
        else:
            print(f"    ✓ Task type is correctly set to 'regression'")

        if not issues_found:
            print(f"    ✓ No obvious issues detected")
    else:
        print(f"  ✗ Train script not found: {train_script}")

    print()

def check_model_weights():
    """Check if pretrained model exists and can be loaded."""
    print("="*80)
    print("5. Checking Pretrained Model")
    print("="*80)

    model_path = "/home/chenx/code/neurostorm_ncc/pt_fmrifound_mae_ratio0.5.ckpt"

    if not os.path.exists(model_path):
        print(f"  ✗ Pretrained model not found: {model_path}")
        print(f"    Model will be trained from scratch (not recommended)")
    else:
        print(f"  ✓ Pretrained model found: {model_path}")
        try:
            ckpt = torch.load(model_path, map_location='cpu')
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
                print(f"    Loaded state_dict with {len(state_dict)} keys")

                # Check for model keys
                model_keys = [k for k in state_dict.keys() if 'model.' in k]
                print(f"    Model keys found: {len(model_keys)}")

                # Check if output_head exists (it shouldn't in pretrained model)
                head_keys = [k for k in state_dict.keys() if 'output_head' in k or 'head' in k]
                if len(head_keys) > 0:
                    print(f"    ⚠ Found {len(head_keys)} head keys - these will be ignored")
                else:
                    print(f"    ✓ No head keys found (expected for pretrained model)")
            else:
                print(f"    ✗ 'state_dict' not found in checkpoint")
        except Exception as e:
            print(f"    ✗ Error loading checkpoint: {e}")

    print()

def main():
    """Run all diagnostic checks."""
    print("\n" + "="*80)
    print("ABIDE Regression Task Diagnostic Report")
    print("="*80 + "\n")

    try:
        check_model_output_head()
    except Exception as e:
        print(f"✗ Error in check_model_output_head: {e}\n")

    try:
        check_data_loading()
    except Exception as e:
        print(f"✗ Error in check_data_loading: {e}\n")

    try:
        check_standardization()
    except Exception as e:
        print(f"✗ Error in check_standardization: {e}\n")

    try:
        check_training_config()
    except Exception as e:
        print(f"✗ Error in check_training_config: {e}\n")

    try:
        check_model_weights()
    except Exception as e:
        print(f"✗ Error in check_model_weights: {e}\n")

    print("="*80)
    print("Common Issues That Can Cause Low R²:")
    print("="*80)
    print("1. ✗ Feature extractor frozen but head too simple (single linear layer)")
    print("2. ✗ Learning rate too small or too large")
    print("3. ✗ Label mismatch (wrong subject IDs mapped to wrong labels)")
    print("4. ✗ Data preprocessing issues (wrong normalization)")
    print("5. ✗ Insufficient training epochs")
    print("6. ✗ Batch size too small causing unstable gradients")
    print("7. ✗ Pretrained model not compatible with current architecture")
    print()
    print("Recommended Actions:")
    print("1. Check tensorboard logs for training loss curve")
    print("2. Print predictions vs true values to see if model is learning")
    print("3. Try unfreezing feature extractor or use deeper regression head")
    print("4. Verify label-sample correspondence manually for a few examples")
    print("="*80)

if __name__ == "__main__":
    main()
