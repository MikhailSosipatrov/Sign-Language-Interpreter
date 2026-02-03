"""
Dataset class for Slovo Russian Sign Language dataset with pre-extracted keypoints
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import random


class SlovoKeypointsDataset(Dataset):
    """
    PyTorch Dataset for Slovo with pre-extracted MediaPipe keypoints

    Args:
        annotations_path: path to annotations.csv
        keypoints_path: path to slovo_mediapipe.json
        split: 'train' or 'test'
        sequence_length: target sequence length (default: 60 frames)
        augment: apply data augmentation (default: False)
    """

    def __init__(
        self,
        annotations_path: str,
        keypoints_path: str,
        split: str = 'train',
        sequence_length: int = 60,
        augment: bool = False
    ):
        self.sequence_length = sequence_length
        self.augment = augment

        # Load annotations
        df = pd.read_csv(annotations_path)

        # Filter by split
        if split == 'train':
            df = df[df['train'] == True]
        else:
            df = df[df['train'] == False]

        # Load keypoints
        print(f"Loading keypoints from {keypoints_path}...")
        with open(keypoints_path, 'r', encoding='utf-8') as f:
            self.keypoints_data = json.load(f)

        # Create class mapping: class_name → index
        unique_classes = sorted(df['text'].unique())
        self.class_to_idx = {name: idx for idx, name in enumerate(unique_classes)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        # Create list of samples
        self.samples = []
        skipped = 0

        for _, row in df.iterrows():
            video_id = row['attachment_id']
            class_name = row['text']

            # Check if keypoints exist
            if video_id in self.keypoints_data:
                keypoints = self.keypoints_data[video_id]
                if len(keypoints) > 0:  # has at least one frame
                    self.samples.append({
                        'video_id': video_id,
                        'class_idx': self.class_to_idx[class_name],
                        'class_name': class_name
                    })
                else:
                    skipped += 1
            else:
                skipped += 1

        print(f"\nLoaded {split} dataset:")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Classes: {len(self.class_to_idx)}")
        print(f"  Skipped: {skipped}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]

        # Load keypoints for this video
        keypoints = self.keypoints_data[sample['video_id']]
        keypoints = np.array(keypoints, dtype=np.float32)

        # keypoints shape: (num_frames, num_keypoints_flat)
        # MediaPipe provides: 21*3 (left hand) + 21*3 (right hand) + 33*3 (pose) = 225

        # Normalize keypoints
        keypoints = self._normalize_keypoints(keypoints)

        # Apply augmentation (if train)
        if self.augment:
            keypoints = self._augment(keypoints)

        # Pad or trim to fixed length
        keypoints = self._pad_or_trim(keypoints, self.sequence_length)

        # Convert to tensor
        x = torch.FloatTensor(keypoints)
        y = sample['class_idx']

        return x, y

    def _normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normalize keypoints by subtracting mean across frames

        Args:
            keypoints: (num_frames, num_features)

        Returns:
            normalized keypoints
        """
        # Replace zeros with NaN for proper mean calculation
        keypoints_copy = keypoints.copy()
        keypoints_copy[keypoints_copy == 0] = np.nan

        # Calculate mean (ignoring NaN)
        mean = np.nanmean(keypoints_copy, axis=0, keepdims=True)

        # Replace NaN back to zeros
        mean = np.nan_to_num(mean)

        # Subtract mean
        normalized = keypoints - mean

        return normalized

    def _pad_or_trim(self, keypoints: np.ndarray, target_length: int) -> np.ndarray:
        """
        Resize sequence to target length

        Args:
            keypoints: (num_frames, num_features)
            target_length: desired number of frames

        Returns:
            resized keypoints: (target_length, num_features)
        """
        current_length = len(keypoints)

        if current_length == target_length:
            return keypoints

        elif current_length > target_length:
            # Downsample: take evenly spaced frames
            indices = np.linspace(0, current_length - 1, target_length, dtype=int)
            return keypoints[indices]

        else:
            # Pad: repeat last frame
            pad_length = target_length - current_length
            last_frame = keypoints[-1:].repeat(pad_length, axis=0)
            return np.vstack([keypoints, last_frame])

    def _augment(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Simple data augmentation

        Args:
            keypoints: (num_frames, num_features)

        Returns:
            augmented keypoints
        """
        # Horizontal flip (50% probability)
        if random.random() > 0.5:
            keypoints = keypoints.copy()
            # Invert X coordinates (every 3rd starting from 0)
            keypoints[:, 0::3] *= -1

        # Gaussian noise (30% probability)
        if random.random() > 0.7:
            noise = np.random.normal(0, 0.02, keypoints.shape)
            keypoints = keypoints + noise

        return keypoints


def create_dataloaders(
    annotations_path: str,
    keypoints_path: str,
    batch_size: int = 32,
    sequence_length: int = 60,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    """
    Create train and test dataloaders

    Args:
        annotations_path: path to annotations.csv
        keypoints_path: path to slovo_mediapipe.json
        batch_size: batch size for dataloaders
        sequence_length: target sequence length
        num_workers: number of workers for data loading

    Returns:
        train_loader, test_loader, idx_to_class mapping
    """
    train_dataset = SlovoKeypointsDataset(
        annotations_path,
        keypoints_path,
        split='train',
        sequence_length=sequence_length,
        augment=True
    )

    test_dataset = SlovoKeypointsDataset(
        annotations_path,
        keypoints_path,
        split='test',
        sequence_length=sequence_length,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, train_dataset.idx_to_class


# Test script
if __name__ == "__main__":
    print("Testing SlovoKeypointsDataset...")

    # Test with dummy paths (replace with actual paths)
    train_loader, test_loader, idx_to_class = create_dataloaders(
        annotations_path='data/raw/annotations.csv',
        keypoints_path='data/raw/slovo_mediapipe.json',
        batch_size=4,
        num_workers=0  # Use 0 for testing on Windows
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Classes: {len(idx_to_class)}")

    # Get one batch
    for x, y in train_loader:
        print(f"\nBatch shape: {x.shape}")  # (batch, sequence, features)
        print(f"Labels shape: {y.shape}")
        print(f"First label: {y[0].item()} = {idx_to_class[y[0].item()]}")
        break

    print("\n✓ Dataset test completed!")
