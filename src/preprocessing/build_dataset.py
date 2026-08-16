"""
Dataset packaging and synthetic sample generator utility for ISL-Speak.
Creates synthetic landmark data for rapid offline testing and packs .npy files into stacked .npz arrays.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.config import (
    SEQ_LEN,
    TOTAL_FEATURES,
    DEFAULT_CLASSES,
    TRAIN_VAL_TEST_SPLIT,
)



def generate_synthetic_landmarks(
    output_dir: str = "data/landmarks",
    labels_csv: str = "data/splits/labels.csv",
    num_samples_per_class: int = 25,
    classes: list = None,
):
    """
    Generates synthetic landmark sequences (seq_len=45, features=258) with distinct class patterns
    for fast offline local pipeline testing without requiring full video downloads.
    """
    if classes is None:
        classes = DEFAULT_CLASSES

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(labels_csv), exist_ok=True)

    records = []
    np.random.seed(42)

    for class_idx, class_name in enumerate(classes):
        base_freq = (class_idx + 1) * 0.5
        phase_shift = class_idx * 0.3

        for i in range(num_samples_per_class):
            t = np.linspace(0, 4 * np.pi, SEQ_LEN)
            # Create synthetic harmonic landmark trajectory
            seq = np.zeros((SEQ_LEN, TOTAL_FEATURES), dtype=np.float32)

            for feat_idx in range(TOTAL_FEATURES):
                freq = base_freq + (feat_idx % 5) * 0.1
                noise = np.random.normal(0, 0.05, size=SEQ_LEN)
                seq[:, feat_idx] = np.sin(freq * t + phase_shift) + noise

            video_id = f"synth_{class_name}_{i:03d}"
            npy_path = os.path.join(output_dir, f"{video_id}.npy")
            np.save(npy_path, seq)

            records.append({
                "video_id": video_id,
                "label": class_name,
                "npy_path": npy_path,
                "frames": SEQ_LEN,
                "features": TOTAL_FEATURES,
            })

    df = pd.DataFrame(records)
    df.to_csv(labels_csv, index=False)
    print(f"Generated {len(records)} synthetic landmark sequences across {len(classes)} classes.")
    return df


def pack_and_split_dataset(
    landmarks_dir: str = "data/landmarks",
    labels_csv: str = "data/splits/labels.csv",
    output_splits_dir: str = "data/splits",
    mapping_json: str = "data/splits/class_index_to_label.json",
):
    """
    Loads all landmark files listed in labels.csv, stacks them into arrays X and y,
    creates stratified train/val/test splits, and saves compressed .npz archives.
    """
    if not os.path.exists(labels_csv):
        print(f"Labels CSV {labels_csv} not found. Generating synthetic dataset first...")
        generate_synthetic_landmarks(output_dir=landmarks_dir, labels_csv=labels_csv)

    df = pd.read_csv(labels_csv)
    labels = sorted(df["label"].unique().tolist())

    label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}
    idx_to_label = {idx: lbl for idx, lbl in enumerate(labels)}

    with open(mapping_json, "w") as f:
        json.dump(idx_to_label, f, indent=2)

    X_list = []
    y_list = []

    for _, row in df.iterrows():
        npy_path = row["npy_path"]
        if os.path.exists(npy_path):
            arr = np.load(npy_path)
            if arr.shape == (SEQ_LEN, TOTAL_FEATURES):
                X_list.append(arr)
                y_list.append(label_to_idx[row["label"]])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")

    # Stratified Train/Val/Test Split (0.70 / 0.15 / 0.15)
    train_ratio, val_ratio, test_ratio = TRAIN_VAL_TEST_SPLIT
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1.0 - train_ratio), stratify=y, random_state=42
    )

    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1.0 - relative_val_ratio), stratify=y_temp, random_state=42
    )

    os.makedirs(output_splits_dir, exist_ok=True)
    np.savez_compressed(os.path.join(output_splits_dir, "X_train.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(output_splits_dir, "X_val.npz"), X=X_val, y=y_val)
    np.savez_compressed(os.path.join(output_splits_dir, "X_test.npz"), X=X_test, y=y_test)

    print(f"Train split: {X_train.shape[0]} samples")
    print(f"Val split:   {X_val.shape[0]} samples")
    print(f"Test split:  {X_test.shape[0]} samples")
    print(f"Saved dataset packages to {output_splits_dir}")
    print(f"Saved class mapping to {mapping_json}")

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    pack_and_split_dataset()
