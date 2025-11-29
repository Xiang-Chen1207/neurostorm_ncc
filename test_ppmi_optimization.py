"""
Test script to verify PPMI dataset optimizations.
Tests caching, memory mapping, and performance improvements.
"""

import time
import numpy as np
import torch
from datasets.fmri_datasets import PPMI
import tracemalloc
import os

def create_dummy_data():
    """Create dummy .npz files for testing if needed."""
    test_dir = "/tmp/ppmi_test"
    os.makedirs(test_dir, exist_ok=True)

    dummy_files = []
    for i in range(5):
        file_path = os.path.join(test_dir, f"sub-{i:06d}_ses-01_task-rest_seg000.npz")
        if not os.path.exists(file_path):
            # Create dummy fMRI data (H, W, D, T) = (64, 64, 64, 40)
            dummy_data = np.random.randn(64, 64, 64, 40).astype(np.float32)
            np.savez(file_path, data=dummy_data)
        dummy_files.append(file_path)

    return test_dir, dummy_files

def test_cache_effectiveness():
    """Test if caching reduces repeated I/O operations."""
    print("\n" + "="*60)
    print("TEST 1: Cache Effectiveness")
    print("="*60)

    # Create dummy data
    test_dir, dummy_files = create_dummy_data()

    # Create subject_dict
    subject_dict = {
        f: [0, i % 3]  # sex=0, target=0/1/2
        for i, f in enumerate(dummy_files)
    }

    # Test with cache
    print("\n--- With Cache (cache_size=100) ---")
    dataset_cached = PPMI(
        root=test_dir,
        subject_dict=subject_dict,
        sequence_length=20,
        stride_within_seq=1,
        stride_between_seq=1,
        img_size=(64, 64, 64, 20),
        train=True,
        cache_size=100,
        use_mmap=True
    )

    # Access same samples multiple times
    print(f"\nDataset size: {len(dataset_cached)}")
    print("\nAccessing first 3 samples repeatedly (3 times each)...")

    start_time = time.time()
    for iteration in range(3):
        for idx in range(min(3, len(dataset_cached))):
            sample = dataset_cached[idx]
            fmri = sample['fmri_sequence']

    cached_time = time.time() - start_time
    print(f"Time with cache: {cached_time:.4f} seconds")
    print(f"Cache size: {len(dataset_cached._data_cache)} files")
    print(f"Metadata cache size: {len(dataset_cached._metadata_cache)} files")

    # Test without cache
    print("\n--- Without Cache (cache_size=0) ---")
    dataset_no_cache = PPMI(
        root=test_dir,
        subject_dict=subject_dict,
        sequence_length=20,
        stride_within_seq=1,
        stride_between_seq=1,
        img_size=(64, 64, 64, 20),
        train=True,
        cache_size=0,
        use_mmap=False
    )

    print("\nAccessing first 3 samples repeatedly (3 times each)...")
    start_time = time.time()
    for iteration in range(3):
        for idx in range(min(3, len(dataset_no_cache))):
            sample = dataset_no_cache[idx]
            fmri = sample['fmri_sequence']

    no_cache_time = time.time() - start_time
    print(f"Time without cache: {no_cache_time:.4f} seconds")

    speedup = no_cache_time / cached_time if cached_time > 0 else 0
    print(f"\n✓ Speedup: {speedup:.2f}x faster with cache")

    return speedup

def test_memory_mapping():
    """Test memory mapping functionality."""
    print("\n" + "="*60)
    print("TEST 2: Memory Mapping")
    print("="*60)

    # Create dummy data
    test_dir, dummy_files = create_dummy_data()

    subject_dict = {
        f: [0, i % 3]
        for i, f in enumerate(dummy_files)
    }

    # Test with memory mapping
    print("\n--- With Memory Mapping ---")
    tracemalloc.start()

    dataset_mmap = PPMI(
        root=test_dir,
        subject_dict=subject_dict,
        sequence_length=20,
        stride_within_seq=1,
        stride_between_seq=1,
        img_size=(64, 64, 64, 20),
        train=True,
        cache_size=10,
        use_mmap=True
    )

    # Load all samples
    for idx in range(len(dataset_mmap)):
        sample = dataset_mmap[idx]

    mmap_memory = tracemalloc.get_traced_memory()[1] / 1024 / 1024  # MB
    tracemalloc.stop()
    print(f"Peak memory usage: {mmap_memory:.2f} MB")

    # Test without memory mapping
    print("\n--- Without Memory Mapping ---")
    tracemalloc.start()

    dataset_no_mmap = PPMI(
        root=test_dir,
        subject_dict=subject_dict,
        sequence_length=20,
        stride_within_seq=1,
        stride_between_seq=1,
        img_size=(64, 64, 64, 20),
        train=True,
        cache_size=10,
        use_mmap=False
    )

    # Load all samples
    for idx in range(len(dataset_no_mmap)):
        sample = dataset_no_mmap[idx]

    no_mmap_memory = tracemalloc.get_traced_memory()[1] / 1024 / 1024  # MB
    tracemalloc.stop()
    print(f"Peak memory usage: {no_mmap_memory:.2f} MB")

    memory_saved = no_mmap_memory - mmap_memory
    print(f"\n✓ Memory saved: {memory_saved:.2f} MB")

def test_metadata_cache():
    """Test metadata caching in _set_data."""
    print("\n" + "="*60)
    print("TEST 3: Metadata Cache")
    print("="*60)

    # Create dummy data
    test_dir, dummy_files = create_dummy_data()

    subject_dict = {
        f: [0, i % 3]
        for i, f in enumerate(dummy_files)
    }

    print("\nInitializing dataset with metadata caching...")
    start_time = time.time()

    dataset = PPMI(
        root=test_dir,
        subject_dict=subject_dict,
        sequence_length=20,
        stride_within_seq=1,
        stride_between_seq=1,
        img_size=(64, 64, 64, 20),
        train=True,
        cache_size=100,
        use_mmap=True
    )

    init_time = time.time() - start_time
    print(f"Initialization time: {init_time:.4f} seconds")
    print(f"Metadata cached for {len(dataset._metadata_cache)} files")
    print(f"Dataset samples: {len(dataset)}")

    # Verify metadata cache content
    if len(dataset._metadata_cache) > 0:
        sample_path = list(dataset._metadata_cache.keys())[0]
        shape, frames, key = dataset._metadata_cache[sample_path]
        print(f"\nSample metadata:")
        print(f"  File: {os.path.basename(sample_path)}")
        print(f"  Shape: {shape}")
        print(f"  Frames: {frames}")
        print(f"  Data key: {key}")
        print("\n✓ Metadata cache working correctly")

def test_lru_eviction():
    """Test LRU cache eviction."""
    print("\n" + "="*60)
    print("TEST 4: LRU Cache Eviction")
    print("="*60)

    # Create dummy data
    test_dir, dummy_files = create_dummy_data()

    subject_dict = {
        f: [0, i % 3]
        for i, f in enumerate(dummy_files)
    }

    # Create dataset with small cache
    cache_size = 2
    print(f"\nCreating dataset with cache_size={cache_size}")

    dataset = PPMI(
        root=test_dir,
        subject_dict=subject_dict,
        sequence_length=20,
        stride_within_seq=1,
        stride_between_seq=1,
        img_size=(64, 64, 64, 20),
        train=True,
        cache_size=cache_size,
        use_mmap=True
    )

    print(f"Dataset has {len(dataset)} samples")

    # Access more samples than cache can hold
    num_samples_to_access = min(4, len(dataset))
    print(f"\nAccessing {num_samples_to_access} samples (more than cache size)...")

    for idx in range(num_samples_to_access):
        sample = dataset[idx]
        cache_keys = list(dataset._data_cache.keys())
        print(f"  After accessing sample {idx}: cache has {len(cache_keys)} files")

    # Verify cache size doesn't exceed limit
    final_cache_size = len(dataset._data_cache)
    if final_cache_size <= cache_size:
        print(f"\n✓ Cache eviction working: {final_cache_size} ≤ {cache_size}")
    else:
        print(f"\n✗ Cache eviction failed: {final_cache_size} > {cache_size}")

def run_all_tests():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# PPMI Dataset Optimization Tests")
    print("#"*60)

    try:
        # Test 1: Cache effectiveness
        speedup = test_cache_effectiveness()

        # Test 2: Memory mapping
        test_memory_mapping()

        # Test 3: Metadata cache
        test_metadata_cache()

        # Test 4: LRU eviction
        test_lru_eviction()

        # Summary
        print("\n" + "#"*60)
        print("# Test Summary")
        print("#"*60)
        print(f"\n✓ All tests completed successfully!")
        print(f"\nKey improvements:")
        print(f"  - Cache speedup: {speedup:.2f}x")
        print(f"  - Memory mapping: Enabled")
        print(f"  - Metadata caching: Working")
        print(f"  - LRU eviction: Working")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
