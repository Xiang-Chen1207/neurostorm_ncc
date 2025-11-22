import os
import torch
from torch.utils.data import Dataset, IterableDataset
import torch.nn.functional as F

import numpy as np
import random
import math
import pandas as pd


def pad_to_96(y):
    background_value = y.flatten()[0]
    y = y.permute(0,4,1,2,3)
    pad_1 = 96 - y.shape[-1]
    pad_2 = 96 - y.shape[-2]
    pad_3 = 96 - y.shape[-3]
    y = torch.nn.functional.pad(y, (math.ceil(pad_1/2), math.floor(pad_1/2), math.ceil(pad_2/2), math.floor(pad_2/2), math.ceil(pad_3/2), math.floor(pad_3/2)), value=background_value)[:,:,:,:,:]
    y = y.permute(0,2,3,4,1)

    return y

import torch
import torch.nn.functional as F

def resize_volume(y, target_size):
    """
    y: 5D tensor [B, H, W, D, T]
    target_size: 4d tuple/list (H', W', D', T)
    """
    current_size = y.shape

    if len(current_size) != 5:
        raise ValueError("Input y must be a 5-dimensional tensor.")
    if len(target_size) != 4:
        raise ValueError("Target size must be a tuple or list of length 4.")

    if current_size[-1] != target_size[-1]:
        raise ValueError(f"y's last dimension {current_size[-1]} and target_size's last dimension {target_size[-1]} must match.")
        
    if current_size[1:4] != tuple(target_size[:3]):
        resized_y = torch.empty(
            (current_size[0],) + tuple(target_size),
            dtype=y.dtype, device=y.device
        )

        for i in range(current_size[0]):
            for t in range(current_size[-1]):
                data_tensor = y[i, :, :, :, t]

                original_dtype = data_tensor.dtype
                resized_tensor = F.interpolate(data_tensor.float().unsqueeze(0).unsqueeze(0), size=tuple(target_size[:3]), mode='trilinear', align_corners=False).squeeze(0).squeeze(0).to(original_dtype)
                resized_y[i, :, :, :, t] = resized_tensor
        return resized_y
    else:
        return y


class BaseDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__()      
        self.register_args(**kwargs)
        self.sample_duration = self.sequence_length * self.stride_within_seq
        self.stride = max(round(self.stride_between_seq * self.sample_duration), 1)
        self.data = self._set_data(self.root, self.subject_dict)

        # import ipdb; ipdb.set_trace()
        # index = 0
        # _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        # y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)
        # y = pad_to_96(y)
        # y = resize_volume(y, self.img_size)
    
    def register_args(self,**kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.kwargs = kwargs
    
    def load_sequence(self, subject_path, start_frame, sample_duration, num_frames=None): 
        if self.contrastive or self.mae:
            num_frames = len(os.listdir(subject_path))
            y = []
            load_fnames = [f'frame_{frame}.pt' for frame in range(start_frame, start_frame+sample_duration, self.stride_within_seq)]
            if self.with_voxel_norm:
                load_fnames += ['voxel_mean.pt', 'voxel_std.pt']

            last_y = None
            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)

                try:
                    y_loaded = torch.load(img_path).unsqueeze(0)
                    y.append(y_loaded)
                    last_y = y_loaded
                except:
                    print('load {} failed'.format(img_path))
                    if last_y is None:
                        y.append(self.previous_last_y)
                        last_y = self.previous_last_y
                    else:
                        y.append(last_y)
            
            self.previous_last_y = y[-1]
            y = torch.cat(y, dim=4)
            
            if self.mae:
                random_y = torch.zeros(1)
            else:
                random_y = []
                
                full_range = np.arange(0, num_frames-sample_duration+1)
                # exclude overlapping sub-sequences within a subject
                exclude_range = np.arange(start_frame-sample_duration, start_frame+sample_duration)
                available_choices = np.setdiff1d(full_range, exclude_range)
                random_start_frame = np.random.choice(available_choices, size=1, replace=False)[0]
                load_fnames = [f'frame_{frame}.pt' for frame in range(random_start_frame, random_start_frame+sample_duration, self.stride_within_seq)]
                if self.with_voxel_norm:
                    load_fnames += ['voxel_mean.pt', 'voxel_std.pt']

                last_y = None
                for fname in load_fnames:
                    img_path = os.path.join(subject_path, fname)

                    try:
                        y_loaded = torch.load(img_path).unsqueeze(0)
                        random_y.append(y_loaded)
                        last_y = y_loaded
                    except:
                        print('load {} failed'.format(img_path))
                        if last_y is None:
                            random_y.append(self.previous_last_y)
                            last_y = self.previous_last_y
                        else:
                            random_y.append(last_y)
                
                self.previous_last_y = y[-1]
                random_y = torch.cat(random_y, dim=4)
            
            return (y, random_y)

        else: # without contrastive learning
            y = []
            if self.shuffle_time_sequence: # shuffle whole sequences
                load_fnames = [f'frame_{frame}.pt' for frame in random.sample(list(range(0, num_frames)), sample_duration // self.stride_within_seq)]
            else:
                load_fnames = [f'frame_{frame}.pt' for frame in range(start_frame, start_frame+sample_duration, self.stride_within_seq)]
            
            if self.with_voxel_norm:
                load_fnames += ['voxel_mean.pt', 'voxel_std.pt']
                
            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)
                y_i = torch.load(img_path).unsqueeze(0)
                y.append(y_i)
            y = torch.cat(y, dim=4)
            return y

    def __len__(self):
        return  len(self.data)

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]

        if self.contrastive or self.mae:
            y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)
            y = pad_to_96(y)
            y = resize_volume(y, self.img_size)

            if self.contrastive:
                rand_y = pad_to_96(rand_y)
                rand_y = resize_volume(rand_y, self.img_size)

            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex
            }
        else:   
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)
            y = pad_to_96(y)
            y = resize_volume(y, self.img_size)

            return {
                "fmri_sequence": y,
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex,
            }

    def _set_data(self, root, subject_dict):
        raise NotImplementedError("Required function")
 

class HCP1200(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')
        for i, subject in enumerate(subject_dict):
            sex,target = subject_dict[subject]
            subject_path = os.path.join(img_root, subject)
            num_frames = len(os.listdir(subject_path))
            session_duration = num_frames - self.sample_duration + 1
            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)

        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)
        return data


class ABCD(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            # subject_name = subject[4:]
            
            subject_path = os.path.join(img_root, subject_name)

            num_frames = len(os.listdir(subject_path))
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)

        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data


class Cobre(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject_path = os.path.join(img_root, subject_name)
            num_frames = len(os.listdir(subject_path))
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
                        
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data


class ADHD200(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject_path = os.path.join(img_root, '{}'.format(subject_name))
            num_frames = len(os.listdir(subject_path))
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
                        
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data
        

class UCLA(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject_path = os.path.join(img_root, '{}'.format(subject_name))
            num_frames = len(os.listdir(subject_path))
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
                        
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data


class HCPEP(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject_path = os.path.join(img_root, '{}'.format(subject_name))
            num_frames = len(os.listdir(subject_path))
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
                        
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data


class GOD(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject_path = os.path.join(img_root, '{}'.format(subject_name))
            num_frames = len(os.listdir(subject_path))
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
                        
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data


class UKB(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            # subject_name = subject[4:]
            
            subject_path = os.path.join(img_root, subject_name)

            num_frames = len(os.listdir(subject_path))
            if num_frames < self.stride:
                import ipdb; ipdb.set_trace()
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)

        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data
    

class HCPTASK(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject_path = os.path.join(img_root, subject_name)

            num_frames = len(os.listdir(subject_path))
            if num_frames < self.stride:
                import ipdb; ipdb.set_trace()
            session_duration = num_frames - self.sample_duration + 1

            # we only use first n frames for task fMRI
            data_tuple = (i, subject_name, subject_path, 0, self.stride, num_frames, target, sex)
            data.append(data_tuple)
                        
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data


class UKB(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            # subject_name = subject[4:]
            
            subject_path = os.path.join(img_root, subject_name)

            num_frames = len(os.listdir(subject_path))
            if num_frames < self.stride:
                import ipdb; ipdb.set_trace()
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)

        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data
    

class MOVIE(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process_tsv(self, path):
        df = pd.read_csv(path, sep='\t')
        result = []
        for _, row in df.iterrows():
            if row['trial_type'] == 'High_cal_food':
                label = 0
            elif row['trial_type'] == 'Low_cal_food':
                label = 1
            elif row['trial_type'] == 'non-food':
                label = 2
            else:
                import ipdb; ipdb.set_trace()
            entry = {
                'start': int(row['onset'] / 0.8),
                'end': int((row['onset'] + row['duration']) / 0.8) + 10,
                'label': label
            }
            result.append(entry)

        result[-1]['end'] -= 10

        return result

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject in enumerate(subject_dict):
            sex, target = subject_dict[subject]
            subject_path = os.path.join(img_root, subject)
            num_frames = len(os.listdir(subject_path))
            session_duration = num_frames - self.sample_duration + 1
            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)

        # label_root = os.path.join(root, 'metadata')

        # for i, subject_name in enumerate(subject_dict):
        #     sex, target = subject_dict[subject_name]
        #     subject_path = os.path.join(img_root, subject_name)
        #     label_path = os.path.join(label_root, '{}-food_events.tsv'.format(subject_name[:-5]))
        #     label = self.process_tsv(label_path)

        #     num_frames = len(os.listdir(subject_path))
        #     if num_frames < self.stride:
        #         import ipdb; ipdb.set_trace()
        #     session_duration = num_frames - self.sample_duration + 1

        #     for j in range(len(label)):
        #         for start_frame in range(label[j]['start'], label[j]['end'], self.stride):
        #             if start_frame + self.stride >= num_frames:
        #                 continue

        #             data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, label[j]['label'], sex)
        #             data.append(data_tuple)
            
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data


class TransDiag(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject in enumerate(subject_dict):
            sex, target = subject_dict[subject]
            subject_path = os.path.join(img_root, subject)
            num_frames = len(os.listdir(subject_path))
            session_duration = num_frames - self.sample_duration + 1
            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)

        if self.train:
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data


class ADNI(BaseDataset):
    """
    ADNI dataset for Alzheimer's Disease classification (AD vs CN).
    Loads .nii.gz files directly and splits them into 20-frame segments.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_sequence(self, subject_path, start_frame, sample_duration, num_frames=None):
        """
        Load a sequence directly from .nii.gz file.
        Args:
            subject_path: Full path to the .nii.gz file
            start_frame: Starting frame index
            sample_duration: Number of frames to load (should be 20)
            num_frames: Total number of frames in the volume (unused for simplicity)
        Returns:
            Tensor of shape (1, H, W, D, 20)
        """
        import nibabel as nib

        # Load the entire 4D volume
        img = nib.load(subject_path)
        data = img.get_fdata()  # Shape: (H, W, D, T)

        # Extract the sequence [start_frame : start_frame + sample_duration]
        sequence = data[:, :, :, start_frame:start_frame + sample_duration]

        # Convert to tensor and add batch dimension
        # Shape: (1, H, W, D, 20)
        y = torch.from_numpy(sequence).float().unsqueeze(0)

        return y

    def _set_data(self, root, subject_dict):
        """
        Set up data list for ADNI dataset.
        Only keeps the first 20 frames from each file.
        Args:
            root: Not used - paths are provided directly in subject_dict
            subject_dict: Dictionary mapping file_path -> [sex, target_label]
        Returns:
            List of tuples: (index, subject_name, file_path, start_frame, stride, num_frames, target, sex)
        """
        data = []
        total_files = len(subject_dict)
        skipped_files = 0
        print(f"Processing {total_files} ADNI files - keeping only first 20 frames from each file...")

        for i, file_path in enumerate(subject_dict):
            sex, target = subject_dict[file_path]

            # Load nii.gz header to get the number of frames (without loading data)
            import nibabel as nib
            try:
                img = nib.load(file_path)
                # Use img.shape instead of get_fdata() to avoid loading data into memory
                num_frames = img.shape[3]  # Time dimension

                # Only keep files with at least 20 frames
                if num_frames < self.sequence_length:
                    print(f"  Skipping {file_path}: only {num_frames} frames (need {self.sequence_length})")
                    skipped_files += 1
                    continue

                # Only use the first 20 frames (start_frame = 0)
                start_frame = 0
                subject_name = os.path.basename(file_path)

                # Data tuple format: (idx, subject_name, file_path, start_frame, sequence_length, num_frames, target, sex)
                data_tuple = (i, subject_name, file_path, start_frame, self.sequence_length, num_frames, target, sex)
                data.append(data_tuple)

                # Print progress every 50 files
                if (i + 1) % 50 == 0 or (i + 1) == total_files:
                    print(f"  Processed {i + 1}/{total_files} files, created {len(data)} samples so far...")

            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                skipped_files += 1
                continue

        print(f"Total: {len(data)} samples created from {total_files} files ({skipped_files} files skipped)")

        if self.train:
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data


class HCP(BaseDataset):
    """
    HCP dataset for sex classification.
    Loads .npz files and extracts the first 20 frames.
    File paths are provided in txt files (hcp_train.txt, hcp_val.txt, hcp_test.txt).
    Labels come from hcp.csv.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_sequence(self, subject_path, start_frame, sample_duration, num_frames=None):
        """
        Load a sequence directly from .npz file.
        Args:
            subject_path: Full path to the .npz file
            start_frame: Starting frame index (should be 0)
            sample_duration: Number of frames to load (should be 20)
            num_frames: Total number of frames in the volume (unused)
        Returns:
            Tensor of shape (1, H, W, D, 20)
        """
        # Load the npz file
        data = np.load(subject_path)

        # The npz file should contain fMRI data
        # Assuming the key is 'data' or similar, we need to check the actual structure
        # Common keys might be 'data', 'arr_0', etc.
        if 'data' in data:
            fmri_data = data['data']
        elif 'arr_0' in data:
            fmri_data = data['arr_0']
        else:
            # Take the first array in the npz file
            key = list(data.keys())[0]
            fmri_data = data[key]

        # Extract the first 20 frames
        # Assuming shape is (H, W, D, T) or (T, H, W, D)
        if fmri_data.shape[-1] >= sample_duration:
            # Last dimension is time
            sequence = fmri_data[:, :, :, start_frame:start_frame + sample_duration]
        elif fmri_data.shape[0] >= sample_duration:
            # First dimension is time, need to transpose
            sequence = fmri_data[start_frame:start_frame + sample_duration, :, :, :]
            sequence = np.transpose(sequence, (1, 2, 3, 0))  # (T, H, W, D) -> (H, W, D, T)
        else:
            raise ValueError(f"npz file {subject_path} has insufficient frames: {fmri_data.shape}")

        # Convert to tensor and add batch dimension
        # Shape: (1, H, W, D, 20)
        y = torch.from_numpy(sequence).float().unsqueeze(0)

        return y

    def _set_data(self, root, subject_dict):
        """
        Set up data list for HCP dataset.
        Only uses the first 20 frames from the FIRST .npz file of each subject.
        Args:
            root: Not used - paths are provided directly in subject_dict
            subject_dict: Dictionary mapping file_path -> [sex, target_label]
        Returns:
            List of tuples: (index, subject_name, file_path, start_frame, stride, num_frames, target, sex)
        """
        data = []
        total_files = len(subject_dict)
        skipped_files = 0
        processed_subjects = set()  # Track which subjects we've already processed
        error_count = 0
        file_not_found_count = 0

        print(f"Processing {total_files} HCP files - keeping only first file per subject, first 20 frames...")

        # Sort file paths to ensure consistent ordering (e.g., _1.npz before _2.npz)
        sorted_file_paths = sorted(subject_dict.keys())

        for i, file_path in enumerate(sorted_file_paths):
            sex, target = subject_dict[file_path]

            try:
                # Extract subject ID from filename
                filename = os.path.basename(file_path)
                # Example: "100307__REST1_LR_hp2000_clean_0000-0199_1.npz" -> "100307"
                subject_id = filename.split('__')[0] if '__' in filename else filename.split('_')[0]

                # Skip if we've already processed this subject
                if subject_id in processed_subjects:
                    skipped_files += 1
                    continue

                # Mark this subject as processed
                processed_subjects.add(subject_id)

                # Check if file exists
                if not os.path.exists(file_path):
                    if file_not_found_count < 5:  # Only print first 5 missing files
                        print(f"  Warning: File not found: {file_path}")
                    file_not_found_count += 1
                    skipped_files += 1
                    continue

                # Load npz file to check number of frames
                npz_data = np.load(file_path)

                # Get the fMRI data array
                if 'data' in npz_data:
                    fmri_data = npz_data['data']
                elif 'arr_0' in npz_data:
                    fmri_data = npz_data['arr_0']
                else:
                    key = list(npz_data.keys())[0]
                    fmri_data = npz_data[key]

                # Determine number of frames (could be first or last dimension)
                if fmri_data.shape[-1] >= self.sequence_length:
                    num_frames = fmri_data.shape[-1]
                elif fmri_data.shape[0] >= self.sequence_length:
                    num_frames = fmri_data.shape[0]
                else:
                    print(f"  Skipping {file_path}: insufficient frames (shape: {fmri_data.shape})")
                    skipped_files += 1
                    continue

                # Only use the first 20 frames (start_frame = 0)
                start_frame = 0
                subject_name = os.path.basename(file_path)

                # Data tuple format: (idx, subject_name, file_path, start_frame, sequence_length, num_frames, target, sex)
                data_tuple = (i, subject_id, file_path, start_frame, self.sequence_length, num_frames, target, sex)
                data.append(data_tuple)

                # Print progress every 1000 files checked (not every 1000 added)
                if (i + 1) % 1000 == 0 or (i + 1) == total_files:
                    print(f"  Checked {i + 1}/{total_files} files, created {len(data)} samples from {len(processed_subjects)} unique subjects...")

            except Exception as e:
                if error_count < 5:  # Only print first 5 errors
                    print(f"Error loading {file_path}: {e}")
                error_count += 1
                skipped_files += 1
                continue

        print(f"Total: {len(data)} samples created from {len(processed_subjects)} unique subjects")
        print(f"  (Skipped {skipped_files} files: {file_not_found_count} not found, {error_count} load errors, {skipped_files - file_not_found_count - error_count} duplicates)")

        if self.train:
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data