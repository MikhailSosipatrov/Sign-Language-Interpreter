"""
Keypoints preprocessing utilities for inference
"""

import numpy as np


def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Normalize keypoints by subtracting mean across frames

    Args:
        keypoints: array of shape (num_frames, num_features)

    Returns:
        normalized keypoints
    """
    keypoints_copy = keypoints.copy()
    keypoints_copy[keypoints_copy == 0] = np.nan

    mean = np.nanmean(keypoints_copy, axis=0, keepdims=True)
    mean = np.nan_to_num(mean)

    normalized = keypoints - mean
    return normalized


def pad_or_trim(keypoints: np.ndarray, target_length: int = 60) -> np.ndarray:
    """
    Resize sequence to target length

    Args:
        keypoints: array of shape (num_frames, num_features)
        target_length: desired number of frames

    Returns:
        resized keypoints of shape (target_length, num_features)
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
