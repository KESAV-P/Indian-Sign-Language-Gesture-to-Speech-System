"""
Data augmentation for ISL landmark sequences.

Applies randomized but physically plausible perturbations to (T, 258) numpy arrays
to artificially multiply the training set. All augmentations preserve the sign's
discriminative structure while adding realistic recording variation.

Augmentation techniques:
  - Time warping (temporal stretch/squeeze)
  - Gaussian coordinate jitter
  - Horizontal mirror (left↔right hand flip)
  - Speed perturbation (frame dropping / duplication)
  - Random frame dropout (simulates occlusion)
  - Magnitude scaling
"""

import numpy as np
from typing import Optional

# Feature layout (must match config.py)
POSE_DIM   = 132   # 33 x 4
LH_DIM     = 63    # 21 x 3
RH_DIM     = 63    # 21 x 3
TOTAL_DIM  = 258


def _pose_slice():
    return slice(0, POSE_DIM)

def _lh_slice():
    return slice(POSE_DIM, POSE_DIM + LH_DIM)

def _rh_slice():
    return slice(POSE_DIM + LH_DIM, TOTAL_DIM)


# ---------------------------------------------------------------------------
# Individual augmentation primitives
# ---------------------------------------------------------------------------

def aug_jitter(seq: np.ndarray, sigma: float = 0.012) -> np.ndarray:
    """Add small Gaussian noise to every coordinate."""
    return seq + np.random.randn(*seq.shape).astype(np.float32) * sigma


def aug_scale(seq: np.ndarray, scale_range=(0.88, 1.12)) -> np.ndarray:
    """Multiply all coordinate values by a scalar near 1.0."""
    s = np.random.uniform(*scale_range)
    return seq * s


def aug_time_warp(seq: np.ndarray) -> np.ndarray:
    """
    Resample the sequence along the time axis using linear interpolation,
    simulating different signing speeds.
    """
    T, F = seq.shape
    # Pick a random start/end inside [0, T) to stretch/compress
    stretch = np.random.uniform(0.75, 1.25)
    new_len = max(10, int(T * stretch))
    src_idx = np.linspace(0, T - 1, new_len)
    warped = np.zeros((new_len, F), dtype=np.float32)
    for i, si in enumerate(src_idx):
        lo = int(si)
        hi = min(lo + 1, T - 1)
        alpha = si - lo
        warped[i] = seq[lo] * (1 - alpha) + seq[hi] * alpha
    # Resample back to T frames
    out_idx = np.linspace(0, new_len - 1, T)
    result = np.zeros((T, F), dtype=np.float32)
    for i, oi in enumerate(out_idx):
        lo = int(oi)
        hi = min(lo + 1, new_len - 1)
        alpha = oi - lo
        result[i] = warped[lo] * (1 - alpha) + warped[hi] * alpha
    return result


def aug_mirror(seq: np.ndarray) -> np.ndarray:
    """
    Horizontal mirror: flip x-coordinates and swap left↔right hands.
    Pose x  → 1 - x  (every 4th value starting at 0)
    LH/RH  → swap the two hand feature blocks and flip their x-coordinates.
    """
    out = seq.copy()
    # Flip pose x coords (index 0 in each group of 4)
    pose = out[:, _pose_slice()].reshape(-1, 33, 4)
    pose[:, :, 0] = 1.0 - pose[:, :, 0]
    out[:, _pose_slice()] = pose.reshape(-1, POSE_DIM)
    # Swap and flip hand blocks
    lh = out[:, _lh_slice()].copy()
    rh = out[:, _rh_slice()].copy()
    # Flip x coords of each hand (every 3rd starting at 0)
    lh_reshaped = lh.reshape(-1, 21, 3)
    rh_reshaped = rh.reshape(-1, 21, 3)
    lh_reshaped[:, :, 0] = 1.0 - lh_reshaped[:, :, 0]
    rh_reshaped[:, :, 0] = 1.0 - rh_reshaped[:, :, 0]
    # Swap: what was left becomes right and vice versa
    out[:, _lh_slice()] = rh_reshaped.reshape(-1, RH_DIM)
    out[:, _rh_slice()] = lh_reshaped.reshape(-1, LH_DIM)
    return out


def aug_frame_dropout(seq: np.ndarray, p: float = 0.10) -> np.ndarray:
    """
    Randomly zero-out p fraction of frames (simulates occlusion / dropped frames).
    Zeroed frames are replaced by the previous valid frame to avoid abrupt gaps.
    """
    out = seq.copy()
    T = seq.shape[0]
    mask = np.random.rand(T) < p
    for t in range(T):
        if mask[t] and t > 0:
            out[t] = out[t - 1]   # repeat previous frame
    return out


def aug_speed_perturb(seq: np.ndarray) -> np.ndarray:
    """
    Simulate signing speed variation by randomly duplicating or dropping frames
    and then resampling back to the original length.
    """
    T, F = seq.shape
    factor = np.random.choice([0.80, 0.90, 1.10, 1.20])
    new_len = max(10, int(T * factor))
    # Simple resample via nearest-neighbour
    src_indices = np.round(np.linspace(0, T - 1, new_len)).astype(int)
    resampled = seq[src_indices]
    # Resample back to T
    back_indices = np.round(np.linspace(0, new_len - 1, T)).astype(int)
    return resampled[back_indices].astype(np.float32)


# ---------------------------------------------------------------------------
# Main augmentation pipeline
# ---------------------------------------------------------------------------

# Each entry: (function, probability)
_AUGMENTATIONS = [
    (aug_jitter,         0.90),
    (aug_scale,          0.70),
    (aug_time_warp,      0.60),
    (aug_frame_dropout,  0.40),
    (aug_speed_perturb,  0.50),
    (aug_mirror,         0.50),
]


def augment_sequence(seq: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Apply a random subset of augmentations to a (T, F) landmark sequence.

    Args:
        seq: Input array of shape (T, TOTAL_FEATURES).
        rng: Optional numpy random Generator for reproducibility.

    Returns:
        Augmented array of shape (T, TOTAL_FEATURES).
    """
    if rng is None:
        rng = np.random.default_rng()

    out = seq.astype(np.float32).copy()
    for fn, prob in _AUGMENTATIONS:
        if rng.random() < prob:
            out = fn(out)

    return out.astype(np.float32)


def augment_dataset(
    X: np.ndarray,
    y: np.ndarray,
    copies_per_sample: int = 9,
    seed: int = 42,
) -> tuple:
    """
    Augment the full dataset by generating `copies_per_sample` augmented versions
    of every sample and appending them to the originals.

    Args:
        X: (N, T, F) float32 array.
        y: (N,) int64 array.
        copies_per_sample: How many augmented copies to generate per original.
        seed: RNG seed for reproducibility.

    Returns:
        (X_aug, y_aug): Arrays with shape ((N * (copies_per_sample+1)), T, F)
                        and ((N * (copies_per_sample+1)),).
    """
    rng = np.random.default_rng(seed)
    N, T, F = X.shape

    X_all = [X]
    y_all = [y]

    for _ in range(copies_per_sample):
        X_copy = np.stack([augment_sequence(X[i], rng) for i in range(N)], axis=0)
        X_all.append(X_copy)
        y_all.append(y.copy())

    X_aug = np.concatenate(X_all, axis=0).astype(np.float32)
    y_aug = np.concatenate(y_all, axis=0).astype(np.int64)

    # Shuffle
    perm = rng.permutation(len(X_aug))
    return X_aug[perm], y_aug[perm]
