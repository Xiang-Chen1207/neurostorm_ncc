#!/usr/bin/env python3
"""
Quick validation script to test MAE implementation components.
This script checks if the MAE components are properly set up.
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    try:
        from datasets.custom_mae_dataset import CustomMAE
        print("✓ CustomMAE dataset imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import CustomMAE: {e}")
        return False
    
    try:
        from models.neurostorm import NeuroSTORMMAE
        print("✓ NeuroSTORMMAE model imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import NeuroSTORMMAE: {e}")
        return False
    
    try:
        from utils.data_module import fMRIDataModule
        print("✓ fMRIDataModule imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import fMRIDataModule: {e}")
        return False
    
    return True

def test_model_instantiation():
    """Test if NeuroSTORMMAE can be instantiated."""
    print("\nTesting model instantiation...")
    
    # Configurable test dimensions - can be modified for different test cases
    TEST_IMG_SIZE = (96, 96, 96, 40)  # (H, W, D, T)
    
    try:
        from models.neurostorm import NeuroSTORMMAE
        import torch
        
        model = NeuroSTORMMAE(
            img_size=TEST_IMG_SIZE,
            in_chans=1,
            embed_dim=36,
            window_size=(4, 4, 4, 4),
            first_window_size=(4, 4, 4, 4),
            patch_size=(4, 4, 4, 5),
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            mask_ratio=0.5,
            spatial_mask='window',
            time_mask='random'
        )
        print("✓ NeuroSTORMMAE model instantiated successfully")
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(1, 1, *TEST_IMG_SIZE)
        try:
            output, loss = model(dummy_input)
            print(f"✓ Forward pass successful, loss shape: {loss.shape if hasattr(loss, 'shape') else 'scalar'}")
            return True
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        return False

def test_data_txt_exists():
    """Test if data.txt template exists."""
    print("\nTesting data.txt file...")
    if os.path.exists('data.txt'):
        print("✓ data.txt file exists")
        with open('data.txt', 'r') as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        if len(lines) == 0:
            print("ℹ data.txt exists but contains no data paths (expected for template)")
        else:
            print(f"✓ data.txt contains {len(lines)} data paths")
        return True
    else:
        print("✗ data.txt file not found")
        return False

def test_training_scripts_exist():
    """Test if training scripts exist and are executable."""
    print("\nTesting training scripts...")
    scripts = ['train_mae_custom.sh', 'train_mae_custom.py']
    all_exist = True
    
    for script in scripts:
        if os.path.exists(script):
            is_executable = os.access(script, os.X_OK)
            if is_executable:
                print(f"✓ {script} exists and is executable")
            else:
                print(f"⚠ {script} exists but is not executable (run: chmod +x {script})")
        else:
            print(f"✗ {script} not found")
            all_exist = False
    
    return all_exist

def test_readme_exists():
    """Test if MAE pretraining README exists."""
    print("\nTesting documentation...")
    if os.path.exists('MAE_PRETRAINING_README.md'):
        print("✓ MAE_PRETRAINING_README.md exists")
        return True
    else:
        print("✗ MAE_PRETRAINING_README.md not found")
        return False

def main():
    print("=" * 60)
    print("MAE Pretraining Implementation Validation")
    print("=" * 60)
    
    results = []
    
    # Run all tests
    results.append(("Import Test", test_imports()))
    results.append(("Model Instantiation", test_model_instantiation()))
    results.append(("Data Template", test_data_txt_exists()))
    results.append(("Training Scripts", test_training_scripts_exist()))
    results.append(("Documentation", test_readme_exists()))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All validation tests passed!")
        print("\nNext steps:")
        print("1. Add your fMRI data paths to data.txt (one path per line)")
        print("2. Run: bash train_mae_custom.sh [batch_size]")
        print("   or:  python train_mae_custom.py --data_txt_path data.txt")
        print("3. Check MAE_PRETRAINING_README.md for detailed instructions")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please review the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
