"""
Dataset packager for ISL-Speak.

Loads real landmark .npy files (extracted by download_include.py from real human signer videos),
stacks them into compressed .npz arrays, and creates stratified train/val/test splits.

NOTE: ALL SYNTHETIC DATA GENERATION HAS BEEN REMOVED.
This script requires real landmark .npy files to already exist (created by download_include.py).
If no data exists, it will print instructions for how to obtain the real INCLUDE dataset.
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
    TRAIN_VAL_TEST_SPLIT,
)


def pack_and_split_dataset(
    landmarks_dir: str = "data/landmarks",
    labels_csv: str = "data/splits/labels.csv",
    output_splits_dir: str = "data/splits",
    mapping_json: str = "data/splits/class_index_to_label.json",
    min_samples_per_class: int = 3,
):
    """
    Loads all real landmark .npy files listed in labels.csv, stacks them into
    arrays X and y, creates stratified train/val/test splits, and saves
    compressed .npz archives.

    Args:
        landmarks_dir: Directory containing .npy landmark files.
        labels_csv: CSV with columns: video_id, label, npy_path, frames, features.
        output_splits_dir: Where to save X_train.npz, X_val.npz, X_test.npz.
        mapping_json: Where to save {class_index: label_name} mapping.
        min_samples_per_class: Drop classes with fewer than this many samples
                                (needed for stratified split to work).
    """
    # ------------------------------------------------------------------
    # Validate that real data exists
    # ------------------------------------------------------------------
    if not os.path.exists(labels_csv):
        _print_no_data_error(labels_csv)
        raise FileNotFoundError(
            f"labels.csv not found at {labels_csv}.\n"
            "Run download_include.py first to download real ISL data."
        )

    df = pd.read_csv(labels_csv)

    if len(df) == 0:
        _print_no_data_error(labels_csv)
        raise ValueError("labels.csv is empty. No real landmark data found.")

    # ------------------------------------------------------------------
    # Filter: drop rows where .npy file is missing
    # ------------------------------------------------------------------
    before = len(df)
    df = df[df["npy_path"].apply(os.path.exists)]
    dropped = before - len(df)
    if dropped > 0:
        print(f"[WARN] {dropped} entries in labels.csv point to missing .npy files — dropped.")
    if len(df) == 0:
        _print_no_data_error(labels_csv)
        raise ValueError("All .npy files are missing. Re-run download_include.py.")

    # ------------------------------------------------------------------
    # Drop classes with too few samples (can't stratify split otherwise)
    # ------------------------------------------------------------------
    class_counts = df["label"].value_counts()
    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    removed_classes = sorted(set(df["label"].unique()) - set(valid_classes))
    if removed_classes:
        print(f"[WARN] Dropping {len(removed_classes)} classes with < {min_samples_per_class} samples:")
        for cls in removed_classes:
            print(f"       {cls}: {class_counts.get(cls, 0)} samples")
        df = df[df["label"].isin(valid_classes)]

    # ------------------------------------------------------------------
    # Build class mapping
    # ------------------------------------------------------------------
    labels = sorted(df["label"].unique().tolist())
    label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}
    idx_to_label = {str(idx): lbl for idx, lbl in enumerate(labels)}

    os.makedirs(output_splits_dir, exist_ok=True)
    with open(mapping_json, "w") as f:
        json.dump(idx_to_label, f, indent=2)

    print(f"\nBuilding dataset from {len(df)} real human ISL landmark sequences...")
    print(f"Classes ({len(labels)}): {labels}\n")

    # ------------------------------------------------------------------
    # Load all .npy files into memory
    # ------------------------------------------------------------------
    X_list = []
    y_list = []
    load_errors = 0

    for _, row in df.iterrows():
        npy_path = row["npy_path"]
        try:
            arr = np.load(npy_path)
            # Validate shape
            if arr.ndim != 2:
                print(f"  [SKIP] {npy_path}: bad ndim {arr.ndim}, expected 2")
                load_errors += 1
                continue
            if arr.shape != (SEQ_LEN, TOTAL_FEATURES):
                # Try to reshape if possible
                if arr.shape[0] == SEQ_LEN and arr.shape[1] != TOTAL_FEATURES:
                    print(f"  [SKIP] {npy_path}: feature mismatch {arr.shape[1]} != {TOTAL_FEATURES}")
                    load_errors += 1
                    continue
                elif arr.shape[1] == TOTAL_FEATURES and arr.shape[0] != SEQ_LEN:
                    # Pad or trim frames
                    from src.preprocessing.extract_landmarks import LandmarkExtractor
                    arr = LandmarkExtractor.normalize_sequence_length(arr, SEQ_LEN)
            X_list.append(arr.astype(np.float32))
            y_list.append(label_to_idx[row["label"]])
        except Exception as e:
            print(f"  [ERROR] Loading {npy_path}: {e}")
            load_errors += 1

    if load_errors > 0:
        print(f"[WARN] {load_errors} files could not be loaded.")

    if not X_list:
        raise RuntimeError("No valid landmark arrays could be loaded. Check your .npy files.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    print(f"Loaded real data shapes: X={X.shape}, y={y.shape}")
    print(f"X value range: min={X.min():.4f}, max={X.max():.4f}, mean={X.mean():.4f}")

    # Sanity check: values should be roughly in [0, 1] for pose/hand coords
    if X.max() > 10.0 or X.min() < -10.0:
        print("[WARN] Feature values appear out of expected [0,1] range — check landmark normalization.")

    # ------------------------------------------------------------------
    # Stratified train/val/test split
    # ------------------------------------------------------------------
    train_ratio, val_ratio, test_ratio = TRAIN_VAL_TEST_SPLIT

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(1.0 - train_ratio),
        stratify=y,
        random_state=42,
    )

    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1.0 - relative_val_ratio),
        stratify=y_temp,
        random_state=42,
    )

    # ------------------------------------------------------------------
    # Save compressed .npz archives
    # ------------------------------------------------------------------
    np.savez_compressed(os.path.join(output_splits_dir, "X_train.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(output_splits_dir, "X_val.npz"),   X=X_val,   y=y_val)
    np.savez_compressed(os.path.join(output_splits_dir, "X_test.npz"),  X=X_test,  y=y_test)

    print(f"\n{'='*55}")
    print(f"  Real ISL Dataset Splits")
    print(f"{'='*55}")
    print(f"  Train : {X_train.shape[0]:>5} samples")
    print(f"  Val   : {X_val.shape[0]:>5} samples")
    print(f"  Test  : {X_test.shape[0]:>5} samples")
    print(f"  Total : {X.shape[0]:>5} samples across {len(labels)} real ISL classes")
    print(f"{'='*55}")
    print(f"  Saved splits to : {output_splits_dir}/")
    print(f"  Class mapping   : {mapping_json}")
    print(f"{'='*55}\n")

    return X_train, y_train, X_val, y_val, X_test, y_test


def _print_no_data_error(labels_csv: str):
    """Print a helpful message when no real data is available."""
    print("\n" + "=" * 65)
    print("  ERROR: No real ISL landmark data found!")
    print("=" * 65)
    print(f"  Expected labels CSV at: {labels_csv}")
    print()
    print("  To get the real AI4Bharat INCLUDE ISL dataset, run:")
    print()
    print("    python src/preprocessing/run_pipeline.py")
    print()
    print("  Or download manually:")
    print("    python src/preprocessing/download_include.py --subset 50")
    print()
    print("  For the full 263-class dataset:")
    print("    python src/preprocessing/download_include.py --subset 263")
    print()
    print("  Dataset info: https://github.com/AI4Bharat/INCLUDE")
    print("  Zenodo:       https://zenodo.org/record/4010759")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pack real ISL landmark .npy files into train/val/test .npz splits."
    )
    parser.add_argument("--landmarks_dir", default="data/landmarks")
    parser.add_argument("--labels_csv", default="data/splits/labels.csv")
    parser.add_argument("--output_dir", default="data/splits")
    parser.add_argument("--mapping_json", default="data/splits/class_index_to_label.json")
    parser.add_argument(
        "--min_samples", type=int, default=3,
        help="Minimum samples per class to include (default: 3)"
    )
    args = parser.parse_args()

    pack_and_split_dataset(
        landmarks_dir=args.landmarks_dir,
        labels_csv=args.labels_csv,
        output_splits_dir=args.output_dir,
        mapping_json=args.mapping_json,
        min_samples_per_class=args.min_samples,
    )
