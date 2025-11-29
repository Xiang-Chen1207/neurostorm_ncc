"""
Test script to verify PPMI data loading works correctly.
Usage: python scripts/ppmi_downstream/test_dataloader.py
"""
import os
import sys

# Add the parent directory to the path
sys.path.insert(0, '/home/user/neurostorm_ncc')

from utils.data_module import fMRIDataModule
from argparse import Namespace

def test_ppmi_dataloader():
    """Test PPMI dataset loading"""

    # Create arguments similar to the training script
    args = Namespace(
        dataset_name='PPMI',
        image_path='/home/user/neurostorm_ncc/data/ppmi',
        img_size=(96, 96, 96, 20),
        sequence_length=20,
        use_contrastive=False,
        contrastive_type='',
        use_mae=False,
        stride_between_seq=1,
        stride_within_seq=1,
        with_voxel_norm=False,
        downstream_task_id=3,
        task_name='group_classification',
        shuffle_time_sequence=False,
        label_scaling_method='',
        pretraining=False,
        dataset_split_num=1,
        train_split=0.7,
        val_split=0.15,
        downstream_task_type='classification',
        batch_size=2,
        num_workers=0,  # Use 0 for testing
        eval_batch_size=2,
    )

    print("=" * 80)
    print("Testing PPMI DataLoader")
    print("=" * 80)

    # Create data module
    print("\n[1] Creating data module...")
    try:
        data_module = fMRIDataModule(**vars(args))
        print("✓ Data module created successfully")
    except Exception as e:
        print(f"✗ Failed to create data module: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check if datasets were created
    print("\n[2] Checking dataset splits...")
    try:
        if hasattr(data_module, 'train_dataset'):
            print(f"✓ Train dataset: {len(data_module.train_dataset)} samples")
        if hasattr(data_module, 'val_dataset'):
            print(f"✓ Validation dataset: {len(data_module.val_dataset)} samples")
        if hasattr(data_module, 'test_dataset'):
            print(f"✓ Test dataset: {len(data_module.test_dataset)} samples")
    except Exception as e:
        print(f"✗ Error checking datasets: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Try to load a batch
    print("\n[3] Testing data loading (first batch)...")
    try:
        train_loader = data_module.train_dataloader()
        print(f"✓ Train dataloader created: {len(train_loader)} batches")

        # Load first batch
        batch = next(iter(train_loader))
        print(f"✓ Successfully loaded first batch")
        print(f"  - fMRI sequence shape: {batch['fmri_sequence'].shape}")
        print(f"  - Target shape: {batch['target'].shape}")
        print(f"  - Target values in batch: {batch['target'].unique()}")
        print(f"  - Subject names: {batch['subject_name']}")

    except Exception as e:
        print(f"✗ Error loading batch: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
    return True

if __name__ == '__main__':
    success = test_ppmi_dataloader()
    sys.exit(0 if success else 1)
