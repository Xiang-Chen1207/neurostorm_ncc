#!/usr/bin/env python
"""
Test script for ADNI dataloader.
This script verifies that the ADNI dataset can be loaded correctly.

Usage:
    python scripts/adni_downstream/test_adni_dataloader.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.data_module import fMRIDataModule
from argparse import Namespace

def test_adni_dataloader():
    """Test the ADNI dataloader with minimal configuration."""

    # Create minimal arguments for testing
    args = Namespace(
        dataset_name='ADNI',
        image_path='/home/chenx/code/NeuroSTORM-main/data',
        img_size=[96, 96, 96, 20],
        sequence_length=20,
        use_contrastive=False,
        contrastive_type='',
        use_mae=False,
        stride_between_seq=1,
        stride_within_seq=1,
        with_voxel_norm=False,
        downstream_task_id=3,
        downstream_task_type='classification',
        task_name='diagnosis',
        shuffle_time_sequence=False,
        label_scaling_method='standardization',
        batch_size=2,
        eval_batch_size=2,
        num_workers=0,  # Set to 0 for testing
        train_split=0.7,
        val_split=0.15,
        dataset_split_num=1,
        bad_subj_path=None,
        limit_training_samples=None,
        pretraining=False
    )

    print("=" * 80)
    print("Testing ADNI Dataloader")
    print("=" * 80)

    try:
        # Create data module
        print("\n[1/4] Creating fMRIDataModule...")
        data_module = fMRIDataModule(**vars(args))

        print("\n[2/4] Checking dataset sizes...")
        print(f"  Train dataset: {len(data_module.train_dataset)} samples")
        print(f"  Val dataset: {len(data_module.val_dataset)} samples")
        print(f"  Test dataset: {len(data_module.test_dataset)} samples")

        print("\n[3/4] Loading a sample from train dataset...")
        sample = data_module.train_dataset[0]

        print("\n[4/4] Sample information:")
        print(f"  fMRI sequence shape: {sample['fmri_sequence'].shape}")
        print(f"  Subject name: {sample['subject_name']}")
        print(f"  Target (label): {sample['target']}")
        print(f"  Sex: {sample['sex']}")
        print(f"  TR (start frame): {sample['TR']}")

        # Verify the shape
        expected_shape = (1, 96, 96, 96, 20)
        actual_shape = tuple(sample['fmri_sequence'].shape)

        if actual_shape == expected_shape:
            print(f"\n✓ Shape verification PASSED: {actual_shape} == {expected_shape}")
        else:
            print(f"\n✗ Shape verification FAILED: {actual_shape} != {expected_shape}")
            return False

        # Load a few more samples to ensure consistency
        print("\n[5/5] Testing multiple samples...")
        for i in range(min(3, len(data_module.train_dataset))):
            sample = data_module.train_dataset[i]
            print(f"  Sample {i}: shape={sample['fmri_sequence'].shape}, label={sample['target']}")

        print("\n" + "=" * 80)
        print("✓ All tests PASSED!")
        print("=" * 80)
        return True

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗ Test FAILED with error:")
        print(f"  {type(e).__name__}: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_adni_dataloader()
    sys.exit(0 if success else 1)