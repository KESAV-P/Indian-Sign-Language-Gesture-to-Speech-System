"""
ISL-Speak Master Pipeline Script.

Runs the complete end-to-end pipeline:
  1. Download real AI4Bharat INCLUDE ISL dataset (human signer videos)
  2. Extract MediaPipe landmarks from real videos
  3. Pack into train/val/test .npz splits
  4. Train BiLSTM or Transformer model on real human data
  5. Report final accuracy

Usage:
    # Full pipeline (recommended — downloads INCLUDE-50, trains LSTM):
    python src/preprocessing/run_pipeline.py

    # Use full 263-class INCLUDE dataset:
    python src/preprocessing/run_pipeline.py --subset 263 --model transformer

    # Quick test with 200 samples only:
    python src/preprocessing/run_pipeline.py --max_samples 200 --epochs 20

    # Skip download if you already have data, just retrain:
    python src/preprocessing/run_pipeline.py --skip_download --model lstm

    # Force re-download and re-extract everything:
    python src/preprocessing/run_pipeline.py --force_redownload
"""

import os
import sys
import argparse
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.config import (
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
)

LANDMARKS_DIR  = os.path.join(PROJECT_ROOT, "data", "landmarks")
SPLITS_DIR     = os.path.join(PROJECT_ROOT, "data", "splits")
LABELS_CSV     = os.path.join(SPLITS_DIR, "labels.csv")
CHECKPOINTS    = os.path.join(PROJECT_ROOT, "checkpoints")
REPORTS_DIR    = os.path.join(PROJECT_ROOT, "reports", "figures")


def print_banner():
    print("\n" + "=" * 65)
    print("  ISL-Speak: Indian Sign Language Gesture-to-Speech")
    print("  Real Human Data Training Pipeline")
    print("=" * 65)
    print("  Dataset  : AI4Bharat INCLUDE (real human signers)")
    print("  License  : CC-BY 4.0")
    print("  Source   : https://github.com/AI4Bharat/INCLUDE")
    print("=" * 65 + "\n")


def step_download(subset: int, max_samples, force_redownload: bool):
    """Step 1: Download real INCLUDE dataset."""
    print("\n[Step 1/3] Downloading real ISL dataset...")
    print("-" * 50)

    from src.preprocessing.download_include import build_real_dataset

    df = build_real_dataset(
        output_landmarks_dir=LANDMARKS_DIR,
        labels_csv=LABELS_CSV,
        include_50_only=(subset == 50),
        max_samples=max_samples,
        force_redownload=force_redownload,
    )

    print(f"✓ Dataset ready: {len(df)} real samples across {df['label'].nunique()} ISL classes")
    return df


def step_pack():
    """Step 2: Pack .npy files into train/val/test splits."""
    print("\n[Step 2/3] Packing real landmarks into train/val/test splits...")
    print("-" * 50)

    from src.preprocessing.build_dataset import pack_and_split_dataset

    results = pack_and_split_dataset(
        landmarks_dir=LANDMARKS_DIR,
        labels_csv=LABELS_CSV,
        output_splits_dir=SPLITS_DIR,
        mapping_json=os.path.join(SPLITS_DIR, "class_index_to_label.json"),
    )

    X_train, y_train, X_val, y_val, X_test, y_test = results
    print(f"✓ Splits saved: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return results


def step_train(model_type: str, epochs: int, batch_size: int):
    """Step 3: Train the model on real ISL data."""
    print(f"\n[Step 3/3] Training {model_type.upper()} model on real human ISL data...")
    print("-" * 50)

    from src.training.train import train_model

    start = time.time()
    model, history, best_val_acc = train_model(
        model_type=model_type,
        splits_dir=SPLITS_DIR,
        checkpoint_dir=CHECKPOINTS,
        figures_dir=REPORTS_DIR,
        epochs=epochs,
        batch_size=batch_size,
        lr=LEARNING_RATE,
    )
    elapsed = time.time() - start

    print(f"\n{'='*65}")
    print(f"  Training Complete on Real Human ISL Data!")
    print(f"{'='*65}")
    print(f"  Best Validation Accuracy : {best_val_acc * 100:.2f}%")
    print(f"  Training Time            : {elapsed / 60:.1f} minutes")
    print(f"  Model saved to           : {CHECKPOINTS}/best_{model_type}_model.pt")
    print(f"  Training curves saved to : {REPORTS_DIR}/")
    print(f"{'='*65}\n")

    # Warn if accuracy is suspiciously perfect (indicates data issue)
    if best_val_acc > 0.99:
        print("[WARNING] Validation accuracy is >99% — this may indicate data leakage")
        print("          or the dataset is too small/simple. Consider using more classes.")
    elif best_val_acc < 0.3:
        print("[NOTE] Accuracy <30% — model is still learning. Try more epochs or more data.")
    else:
        print("[OK] Accuracy looks reasonable for real human sign language data.")

    return model, history, best_val_acc


def main():
    parser = argparse.ArgumentParser(
        description="ISL-Speak: Download real INCLUDE dataset and train gesture classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--subset", type=int, choices=[50, 263], default=50,
        help="INCLUDE-50 (50 classes, faster) or full INCLUDE (263 classes). Default: 50"
    )
    parser.add_argument(
        "--model", type=str, choices=["lstm", "transformer"], default="lstm",
        help="Model architecture to train. Default: lstm"
    )
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS,
        help=f"Training epochs. Default: {NUM_EPOCHS}"
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help=f"Batch size. Default: {BATCH_SIZE}"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit total samples downloaded (for quick testing). Default: no limit"
    )
    parser.add_argument(
        "--skip_download", action="store_true",
        help="Skip download step — use existing data in data/landmarks/"
    )
    parser.add_argument(
        "--skip_pack", action="store_true",
        help="Skip packing step — use existing .npz splits in data/splits/"
    )
    parser.add_argument(
        "--force_redownload", action="store_true",
        help="Force re-download and re-extraction even if files already exist"
    )
    parser.add_argument(
        "--train_only", action="store_true",
        help="Skip download and packing, only run training (data must already exist)"
    )
    args = parser.parse_args()

    print_banner()
    t_start = time.time()

    # ── Download ──────────────────────────────────────────────────────
    if not args.skip_download and not args.train_only:
        step_download(
            subset=args.subset,
            max_samples=args.max_samples,
            force_redownload=args.force_redownload,
        )
    else:
        print("[Step 1/3] Skipping download — using existing data.")

    # ── Pack ──────────────────────────────────────────────────────────
    if not args.skip_pack and not args.train_only:
        step_pack()
    else:
        print("[Step 2/3] Skipping pack — using existing .npz splits.")

    # ── Train ─────────────────────────────────────────────────────────
    step_train(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    total_time = (time.time() - t_start) / 60
    print(f"Total pipeline time: {total_time:.1f} minutes")


if __name__ == "__main__":
    main()
