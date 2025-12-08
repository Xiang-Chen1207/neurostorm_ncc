"""
Custom dataset for MAE pretraining that loads data paths from a text file.
"""
import os
import torch
import numpy as np
import nibabel as nib
from .fmri_datasets import BaseDataset


class CustomMAE(BaseDataset):
    """
    Custom dataset for MAE pretraining that loads data paths from a text file.
    Each line in the text file should contain the path to a data file.
    Supports .npz, .npy, .pt, and .nii.gz formats.
    Expected data shape: (96, 96, 96, 200) or similar 4D fMRI volumes.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _set_data(self, root, subject_dict):
        """
        Load data paths from a text file specified in subject_dict.
        
        Args:
            root: Root directory path (can be used as prefix if paths in txt are relative)
            subject_dict: Dictionary containing 'data_txt_path' key pointing to the text file
        
        Returns:
            List of data tuples in format: (idx, subject_id, file_path, start_frame, sequence_length, num_frames, target, sex)
        """
        data = []
        
        # Get the path to data.txt file
        if isinstance(subject_dict, dict) and 'data_txt_path' in subject_dict:
            data_txt_path = subject_dict['data_txt_path']
        else:
            # Default to data.txt in current directory
            data_txt_path = 'data.txt'
        
        if not os.path.exists(data_txt_path):
            print(f"Warning: {data_txt_path} not found. Creating empty dataset.")
            return data
        
        print(f"Loading data paths from: {data_txt_path}")
        
        # Read all file paths from data.txt
        with open(data_txt_path, 'r') as f:
            file_paths = [line.strip() for line in f if line.strip()]
        
        total_files = len(file_paths)
        print(f"Found {total_files} file paths in {data_txt_path}")
        
        error_count = 0
        skipped_files = 0
        file_not_found_count = 0
        
        for i, file_path in enumerate(file_paths):
            try:
                # Handle relative paths
                if not os.path.isabs(file_path) and root:
                    file_path = os.path.join(root, file_path)
                
                # Check if file exists
                if not os.path.exists(file_path):
                    if error_count < 5:
                        print(f"  File not found: {file_path}")
                    file_not_found_count += 1
                    skipped_files += 1
                    continue
                
                # Extract subject ID from filename
                subject_id = os.path.splitext(os.path.basename(file_path))[0]
                
                # Load data to check shape and number of frames
                num_frames = None
                
                if file_path.endswith('.npz'):
                    npz_data = np.load(file_path)
                    # Try common keys
                    if 'data' in npz_data:
                        fmri_data = npz_data['data']
                    elif 'arr_0' in npz_data:
                        fmri_data = npz_data['arr_0']
                    else:
                        key = list(npz_data.keys())[0]
                        fmri_data = npz_data[key]
                    
                    # Determine number of frames (last dimension should be time)
                    if len(fmri_data.shape) == 4:
                        num_frames = fmri_data.shape[-1]
                    else:
                        if error_count < 5:
                            print(f"  Skipping {file_path}: unexpected shape {fmri_data.shape}, expected 4D")
                        error_count += 1
                        skipped_files += 1
                        continue
                        
                elif file_path.endswith('.npy'):
                    fmri_data = np.load(file_path)
                    if len(fmri_data.shape) == 4:
                        num_frames = fmri_data.shape[-1]
                    else:
                        if error_count < 5:
                            print(f"  Skipping {file_path}: unexpected shape {fmri_data.shape}, expected 4D")
                        error_count += 1
                        skipped_files += 1
                        continue
                        
                elif file_path.endswith('.pt'):
                    fmri_data = torch.load(file_path)
                    if isinstance(fmri_data, torch.Tensor):
                        fmri_data = fmri_data.numpy()
                    if len(fmri_data.shape) == 4:
                        num_frames = fmri_data.shape[-1]
                    else:
                        if error_count < 5:
                            print(f"  Skipping {file_path}: unexpected shape {fmri_data.shape}, expected 4D")
                        error_count += 1
                        skipped_files += 1
                        continue
                        
                elif file_path.endswith('.nii.gz') or file_path.endswith('.nii'):
                    nii_img = nib.load(file_path)
                    fmri_data = nii_img.get_fdata()
                    if len(fmri_data.shape) == 4:
                        num_frames = fmri_data.shape[-1]
                    else:
                        if error_count < 5:
                            print(f"  Skipping {file_path}: unexpected shape {fmri_data.shape}, expected 4D")
                        error_count += 1
                        skipped_files += 1
                        continue
                else:
                    if error_count < 5:
                        print(f"  Skipping {file_path}: unsupported file format")
                    error_count += 1
                    skipped_files += 1
                    continue
                
                # Check if sufficient frames for sequence_length
                if num_frames < self.sequence_length:
                    if error_count < 5:
                        print(f"  Skipping {file_path}: insufficient frames ({num_frames} < {self.sequence_length})")
                    error_count += 1
                    skipped_files += 1
                    continue
                
                # Create multiple samples from the same file if it has many frames
                session_duration = num_frames - self.sample_duration + 1
                
                # For MAE pretraining, we typically use overlapping or non-overlapping windows
                for start_frame in range(0, session_duration, self.stride):
                    # Dummy target and sex values for pretraining (not used in MAE)
                    target = 0
                    sex = 0
                    
                    # Data tuple format: (idx, subject_id, file_path, start_frame, sequence_length, num_frames, target, sex)
                    data_tuple = (len(data), subject_id, file_path, start_frame, self.sequence_length, num_frames, target, sex)
                    data.append(data_tuple)
                
                # Print progress every 50 files checked
                if (i + 1) % 50 == 0 or (i + 1) == total_files:
                    print(f"  Processed {i + 1}/{total_files} files, created {len(data)} samples...")
                    
            except Exception as e:
                if error_count < 5:
                    print(f"  Error loading {file_path}: {e}")
                error_count += 1
                skipped_files += 1
                continue
        
        print(f"Dataset loaded: {len(data)} samples from {total_files - skipped_files} valid files")
        if skipped_files > 0:
            print(f"  Skipped {skipped_files} files: {file_not_found_count} not found, {error_count - file_not_found_count} load errors")
        
        if self.train:
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)
        
        return data
    
    def load_sequence(self, subject_path, start_frame, sample_duration, num_frames=None):
        """
        Load a sequence from a single file (different from base class which loads from directories).
        
        Args:
            subject_path: Path to the data file (not a directory)
            start_frame: Starting frame index
            sample_duration: Number of frames to load
            num_frames: Total number of frames in the file
        
        Returns:
            Tuple of (y, random_y) for contrastive learning or MAE
        """
        # Load the full data file
        if subject_path.endswith('.npz'):
            npz_data = np.load(subject_path)
            if 'data' in npz_data:
                fmri_data = npz_data['data']
            elif 'arr_0' in npz_data:
                fmri_data = npz_data['arr_0']
            else:
                key = list(npz_data.keys())[0]
                fmri_data = npz_data[key]
                
        elif subject_path.endswith('.npy'):
            fmri_data = np.load(subject_path)
            
        elif subject_path.endswith('.pt'):
            fmri_data = torch.load(subject_path)
            if isinstance(fmri_data, torch.Tensor):
                fmri_data = fmri_data.numpy()
                
        elif subject_path.endswith('.nii.gz') or subject_path.endswith('.nii'):
            nii_img = nib.load(subject_path)
            fmri_data = nii_img.get_fdata()
        else:
            raise ValueError(f"Unsupported file format: {subject_path}")
        
        # Extract the sequence we need: shape should be (H, W, D, T)
        # We need frames from start_frame to start_frame + sample_duration
        end_frame = start_frame + sample_duration
        
        # Handle different stride_within_seq
        if self.stride_within_seq > 1:
            frame_indices = list(range(start_frame, end_frame, self.stride_within_seq))
        else:
            frame_indices = list(range(start_frame, end_frame))
        
        # Extract the frames
        y = fmri_data[..., frame_indices]  # Shape: (H, W, D, T')
        
        # Convert to torch tensor and add batch and channel dimensions
        y = torch.from_numpy(y).float()
        y = y.unsqueeze(0)  # Add batch dim: (1, H, W, D, T')
        
        if self.mae:
            # For MAE, we don't need a second augmented view, just return a dummy
            random_y = torch.zeros(1)
        elif self.contrastive:
            # For contrastive learning, create a second view from a different time window
            if num_frames is None:
                num_frames = fmri_data.shape[-1]
            
            full_range = np.arange(0, num_frames - sample_duration + 1)
            exclude_range = np.arange(start_frame - sample_duration, start_frame + sample_duration)
            available_choices = np.setdiff1d(full_range, exclude_range)
            
            if len(available_choices) > 0:
                random_start_frame = np.random.choice(available_choices, size=1, replace=False)[0]
                random_end_frame = random_start_frame + sample_duration
                
                if self.stride_within_seq > 1:
                    random_frame_indices = list(range(random_start_frame, random_end_frame, self.stride_within_seq))
                else:
                    random_frame_indices = list(range(random_start_frame, random_end_frame))
                
                random_y = fmri_data[..., random_frame_indices]
                random_y = torch.from_numpy(random_y).float()
                random_y = random_y.unsqueeze(0)
            else:
                # Fallback: use the same sequence
                random_y = y.clone()
        else:
            random_y = None
        
        if self.contrastive or self.mae:
            return (y, random_y)
        else:
            return y
