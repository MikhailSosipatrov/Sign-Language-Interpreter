"""
Dataset class for Slovo Russian Sign Language dataset with pre-extracted keypoints.
"""

import json
import random
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def _read_annotations(annotations_path: str) -> pd.DataFrame:
    """Read annotations robustly (Slovo uses tab-separated CSV)."""
    try:
        df = pd.read_csv(annotations_path, sep="	", encoding="utf-8")
    except Exception:
        # Fallback for malformed rows/separators.
        df = pd.read_csv(annotations_path, sep=None, engine="python", encoding="utf-8")

    required_cols = {"attachment_id", "text", "train"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"annotations file is missing required columns: {sorted(missing)}")

    if df["train"].dtype != bool:
        df["train"] = df["train"].astype(str).str.lower().isin(["true", "1", "yes"])

    df["attachment_id"] = df["attachment_id"].astype(str)
    return df


HAND_ORDER = ("hand 1", "hand 2", "hand 3")
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
DEFAULT_FEATURE_DIM = len(HAND_ORDER) * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _frame_dict_to_vector(frame: Dict[str, Any]) -> np.ndarray:
    """
    Convert one frame dict to a fixed-size vector.

    Expected frame format (current Slovo dump):
      {
        "hand 1": [{"x":..., "y":..., "z":...}, ... 21 items ...],
        "hand 2": [...],
        "hand 3": [...]
      }
    Missing hands/landmarks are zero-padded.
    """
    vector = np.zeros(DEFAULT_FEATURE_DIM, dtype=np.float32)
    offset = 0

    for hand_name in HAND_ORDER:
        landmarks = frame.get(hand_name, [])
        if isinstance(landmarks, list):
            for idx, lm in enumerate(landmarks[:LANDMARKS_PER_HAND]):
                if isinstance(lm, dict):
                    base = offset + idx * COORDS_PER_LANDMARK
                    vector[base] = _safe_float(lm.get("x", 0.0))
                    vector[base + 1] = _safe_float(lm.get("y", 0.0))
                    vector[base + 2] = _safe_float(lm.get("z", 0.0))
        offset += LANDMARKS_PER_HAND * COORDS_PER_LANDMARK

    return vector


def _keypoints_to_array(raw_keypoints: Any) -> np.ndarray:
    """
    Convert different keypoint representations to (num_frames, num_features) float32 array.
    """
    if isinstance(raw_keypoints, list):
        if not raw_keypoints:
            return np.zeros((1, DEFAULT_FEATURE_DIM), dtype=np.float32)

        first = raw_keypoints[0]
        if isinstance(first, dict):
            frames = [_frame_dict_to_vector(frame) for frame in raw_keypoints]
            return np.stack(frames, axis=0).astype(np.float32, copy=False)

        arr = np.array(raw_keypoints, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr

    if isinstance(raw_keypoints, dict):
        # Fallback: dict keyed by frame index -> frame content
        ordered_frames = [raw_keypoints[k] for k in sorted(raw_keypoints.keys())]
        if ordered_frames and isinstance(ordered_frames[0], dict):
            frames = [_frame_dict_to_vector(frame) for frame in ordered_frames]
            return np.stack(frames, axis=0).astype(np.float32, copy=False)

    raise TypeError(f"Unsupported keypoints format: {type(raw_keypoints).__name__}")


class SlovoKeypointsDataset(Dataset):
    """
    PyTorch Dataset for Slovo with pre-extracted MediaPipe keypoints.

    Args:
        annotations_path: path to annotations.csv (TSV)
        keypoints_path: path to slovo_mediapipe.json (used if keypoints_data is None)
        split: 'train' or 'test'
        sequence_length: target sequence length (default: 60 frames)
        augment: apply data augmentation (default: False)
        keypoints_data: optional preloaded keypoints dictionary
        class_to_idx: optional shared class mapping for train/test
    """

    def __init__(
        self,
        annotations_path: str,
        keypoints_path: str | None = None,
        split: str = "train",
        sequence_length: int = 60,
        augment: bool = False,
        keypoints_data: Dict[str, list] | None = None,
        class_to_idx: Dict[str, int] | None = None,
    ):
        self.sequence_length = sequence_length
        self.augment = augment

        df = _read_annotations(annotations_path)

        if split == "train":
            df = df[df["train"] == True]
        else:
            df = df[df["train"] == False]

        if keypoints_data is None:
            if keypoints_path is None:
                raise ValueError("keypoints_path must be provided when keypoints_data is None")
            print(f"Loading keypoints from {keypoints_path}...")
            with open(keypoints_path, "r", encoding="utf-8") as f:
                self.keypoints_data = json.load(f)
        else:
            self.keypoints_data = keypoints_data

        if class_to_idx is None:
            unique_classes = sorted(df["text"].dropna().unique())
            self.class_to_idx = {name: idx for idx, name in enumerate(unique_classes)}
        else:
            self.class_to_idx = class_to_idx
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        self.samples = []
        skipped = 0

        for _, row in df.iterrows():
            video_id = str(row["attachment_id"])
            class_name = row["text"]

            if video_id in self.keypoints_data and class_name in self.class_to_idx:
                keypoints = self.keypoints_data[video_id]
                if len(keypoints) > 0:
                    self.samples.append(
                        {
                            "video_id": video_id,
                            "class_idx": self.class_to_idx[class_name],
                            "class_name": class_name,
                        }
                    )
                else:
                    skipped += 1
            else:
                skipped += 1

        print("")
        print(f"Loaded {split} dataset:")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Classes: {len(self.class_to_idx)}")
        print(f"  Skipped: {skipped}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]

        raw_keypoints = self.keypoints_data[sample["video_id"]]
        keypoints = _keypoints_to_array(raw_keypoints)

        keypoints = self._normalize_keypoints(keypoints)

        if self.augment:
            keypoints = self._augment(keypoints)

        keypoints = self._pad_or_trim(keypoints, self.sequence_length)

        x = torch.FloatTensor(keypoints)
        y = sample["class_idx"]

        return x, y

    def _normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        # Treat zeros as missing points and compute mean without warnings.
        mask = keypoints != 0
        counts = mask.sum(axis=0, keepdims=True)
        sums = np.where(mask, keypoints, 0.0).sum(axis=0, keepdims=True)
        mean = np.divide(
            sums,
            counts,
            out=np.zeros_like(sums, dtype=np.float32),
            where=counts > 0,
        )
        return keypoints - mean

    def _pad_or_trim(self, keypoints: np.ndarray, target_length: int) -> np.ndarray:
        current_length = len(keypoints)

        if current_length == target_length:
            return keypoints
        if current_length > target_length:
            indices = np.linspace(0, current_length - 1, target_length, dtype=int)
            return keypoints[indices]

        pad_length = target_length - current_length
        last_frame = keypoints[-1:].repeat(pad_length, axis=0)
        return np.vstack([keypoints, last_frame])

    def _augment(self, keypoints: np.ndarray) -> np.ndarray:
        if random.random() > 0.5:
            keypoints = keypoints.copy()
            keypoints[:, 0::3] *= -1

        if random.random() > 0.7:
            noise = np.random.normal(0, 0.02, keypoints.shape)
            keypoints = keypoints + noise

        return keypoints


def create_dataloaders(
    annotations_path: str,
    keypoints_path: str,
    batch_size: int = 32,
    sequence_length: int = 60,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    """
    Create train and test dataloaders.

    Keypoints JSON is loaded once and reused for both splits to avoid 2x RAM usage.
    """
    full_df = _read_annotations(annotations_path)
    all_classes = sorted(full_df["text"].dropna().unique())
    class_to_idx = {name: idx for idx, name in enumerate(all_classes)}

    print(f"Loading keypoints from {keypoints_path}...")
    with open(keypoints_path, "r", encoding="utf-8") as f:
        keypoints_data = json.load(f)

    train_dataset = SlovoKeypointsDataset(
        annotations_path=annotations_path,
        split="train",
        sequence_length=sequence_length,
        augment=True,
        keypoints_data=keypoints_data,
        class_to_idx=class_to_idx,
    )

    test_dataset = SlovoKeypointsDataset(
        annotations_path=annotations_path,
        split="test",
        sequence_length=sequence_length,
        augment=False,
        keypoints_data=keypoints_data,
        class_to_idx=class_to_idx,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader, train_dataset.idx_to_class


if __name__ == "__main__":
    print("Testing SlovoKeypointsDataset...")

    train_loader, test_loader, idx_to_class = create_dataloaders(
        annotations_path="data/raw/annotations.csv",
        keypoints_path="data/raw/slovo_mediapipe.json",
        batch_size=4,
        num_workers=0,
    )

    print("")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Classes: {len(idx_to_class)}")

    for x, y in train_loader:
        print("")
        print(f"Batch shape: {x.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"First label: {y[0].item()} = {idx_to_class[y[0].item()]}")
        break

    print("")
    print("Dataset test completed!")
