#!/usr/bin/env python3
"""
INCLUDE Dataset Real Video Downloader & Landmark Extractor.

Downloads specific category ZIP files from the AI4Bharat INCLUDE Zenodo record,
extracts real human signer videos, runs MediaPipe landmark extraction, and saves
genuine .npy landmark files.

This replaces ALL synthetic data generation.

Usage:
    python src/preprocessing/fetch_include_real.py
    python src/preprocessing/fetch_include_real.py --categories Greetings Days_and_Time
    python src/preprocessing/fetch_include_real.py --max_per_class 20 --categories Electronics
"""

import os
import sys
import re
import zipfile
import tempfile
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.config import SEQ_LEN, TOTAL_FEATURES
from src.preprocessing.extract_landmarks import LandmarkExtractor

ZENODO_BASE = "https://zenodo.org/api/records/4010759/files"
LANDMARKS_DIR = os.path.join(PROJECT_ROOT, "data", "landmarks")
RAW_VIDEO_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "include")
LABELS_CSV    = os.path.join(PROJECT_ROOT, "data", "splits", "labels.csv")

# -------------------------------------------------------------------
# Available Zenodo categories with their ZIP files (sorted by total size).
# We prioritize smaller archives and INCLUDE-50 relevant categories.
# -------------------------------------------------------------------
ZENODO_CATEGORY_ZIPS = {
    # ~1.7 GB total (2 parts) — High priority for daily language
    "Days_and_Time": [
        "Days_and_Time_3of3.zip",   # 853 MB
    ],
    # ~824 MB — Electronics category
    "Electronics": [
        "Electronics_2of2.zip",     # 824 MB
    ],
    # ~924 MB — Pronouns
    "Pronouns": [
        "Pronouns_2of2.zip",        # 946 MB
    ],
    # ~873 MB — Home/daily life
    "Home": [
        "Home_4of4.zip",            # 873 MB
    ],
    # ~1252 MB — Seasons
    "Seasons": [
        "Seasons_1of1.zip",         # 1252 MB
    ],
    # ~2.7 GB total for Greetings (we already have local Greetings data)
    "Greetings": [
        "Greetings_2of2.zip",       # 1208 MB (Greetings_1of2 is 1573 MB, skip it)
    ],
    # Animals
    "Animals": [
        "Animals_2of2.zip",         # 1069 MB
    ],
    # People
    "People": [
        "People_2of5.zip",          # 1251 MB
    ],
}

# Default: download these smaller categories first (gives good class diversity)
DEFAULT_CATEGORIES = ["Days_and_Time", "Electronics", "Pronouns", "Home", "Seasons"]


def clean_label(raw_label: str) -> str:
    """Strip leading numeric prefix (e.g. '48. Hello' -> 'Hello')."""
    cleaned = re.sub(r"^\d+[\.\-\_\s]+", "", raw_label).strip()
    return cleaned.title() if cleaned else raw_label.strip()


def download_zip_stream(url: str, dest_path: str, category: str) -> bool:
    """Download a ZIP file with a progress bar."""
    print(f"\n→ Downloading {category} from Zenodo...")
    print(f"  URL: {url}")
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total_size = int(resp.headers.get("content-length", 0))
        
        with open(dest_path, "wb") as f, tqdm(
            desc=f"  {category}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        
        size_mb = os.path.getsize(dest_path) / 1e6
        print(f"  ✓ Downloaded {size_mb:.1f} MB → {dest_path}")
        return True
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def extract_landmarks_from_zip(
    zip_path: str,
    output_dir: str,
    extractor: LandmarkExtractor,
    max_per_class: Optional[int] = None,
    existing_labels: Optional[set] = None,
) -> List[Dict]:
    """
    Open a Zenodo ZIP, iterate video files, extract landmarks on-the-fly.
    Avoids extracting the full ZIP to disk (saves space).

    ZIP structure: <Category>/<word_class>/<video.MOV>
    """
    records = []
    existing_labels = existing_labels or set()

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()
        video_names = [n for n in all_names if n.lower().endswith((".mov", ".mp4", ".avi", ".mkv"))]
        print(f"  Found {len(video_names)} videos in {os.path.basename(zip_path)}")

        # Count per class to respect max_per_class limit
        class_counts: Dict[str, int] = {}

        for video_name in tqdm(video_names, desc="  Extracting landmarks", unit="video"):
            parts = Path(video_name).parts
            # Parts: ('Category', 'word_class', 'video.MOV')
            if len(parts) < 2:
                continue

            raw_label = parts[-2]  # e.g. '48. Hello'
            label = clean_label(raw_label)
            video_stem = Path(video_name).stem

            # Skip if max_per_class reached
            if max_per_class and class_counts.get(label, 0) >= max_per_class:
                continue

            safe_label = label.replace(" ", "_")
            npy_filename = f"include_{safe_label}_{video_stem}.npy"
            npy_save_path = os.path.join(output_dir, npy_filename)

            # Skip if already extracted
            if os.path.exists(npy_save_path):
                records.append({
                    "video_id": video_stem,
                    "label": label,
                    "npy_path": npy_save_path,
                    "frames": SEQ_LEN,
                    "features": TOTAL_FEATURES,
                    "source": "include_zenodo",
                })
                class_counts[label] = class_counts.get(label, 0) + 1
                continue

            # Extract single video to temp file, process, delete immediately
            try:
                with tempfile.NamedTemporaryFile(suffix=".mov", delete=False) as tmp:
                    tmp.write(zf.read(video_name))
                    tmp_path = tmp.name

                landmarks = extractor.process_video(tmp_path, target_seq_len=SEQ_LEN)
                os.unlink(tmp_path)

                np.save(npy_save_path, landmarks)
                records.append({
                    "video_id": video_stem,
                    "label": label,
                    "npy_path": npy_save_path,
                    "frames": SEQ_LEN,
                    "features": TOTAL_FEATURES,
                    "source": "include_zenodo",
                })
                class_counts[label] = class_counts.get(label, 0) + 1

            except Exception as e:
                print(f"\n  [ERROR] {video_name}: {e}")
                if "tmp_path" in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    unique_classes = set(r["label"] for r in records)
    print(f"  ✓ Extracted {len(records)} landmark files across {len(unique_classes)} classes")
    return records


def fetch_and_extract(
    categories: List[str] = None,
    output_dir: str = LANDMARKS_DIR,
    labels_csv: str = LABELS_CSV,
    max_per_class: Optional[int] = None,
    keep_zips: bool = False,
) -> pd.DataFrame:
    """
    Main function: download Zenodo ZIPs for given categories, extract real
    human landmarks, save .npy files, and update labels.csv.

    Args:
        categories: List of INCLUDE categories to download. Defaults to DEFAULT_CATEGORIES.
        output_dir: Where to save .npy files.
        labels_csv: CSV to update with new records.
        max_per_class: Limit samples per class (None = no limit).
        keep_zips: Keep downloaded ZIP files after extraction (default: delete to save space).
    """
    if categories is None:
        categories = DEFAULT_CATEGORIES

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(labels_csv), exist_ok=True)
    os.makedirs(RAW_VIDEO_DIR, exist_ok=True)

    # Load existing records if any
    all_records: List[Dict] = []
    if os.path.exists(labels_csv):
        existing_df = pd.read_csv(labels_csv)
        # Only keep rows where the .npy file actually exists
        existing_df = existing_df[existing_df["npy_path"].apply(os.path.exists)]
        all_records = existing_df.to_dict("records")
        print(f"Loaded {len(all_records)} existing real landmark records from {labels_csv}")

    extractor = LandmarkExtractor()
    zip_dir = RAW_VIDEO_DIR

    for category in categories:
        if category not in ZENODO_CATEGORY_ZIPS:
            print(f"[WARN] Unknown category '{category}'. Available: {list(ZENODO_CATEGORY_ZIPS.keys())}")
            continue

        zip_files = ZENODO_CATEGORY_ZIPS[category]
        print(f"\n{'='*55}")
        print(f"  Category: {category} ({len(zip_files)} ZIP files)")
        print(f"{'='*55}")

        for zip_filename in zip_files:
            zip_local_path = os.path.join(zip_dir, zip_filename)
            zip_url = f"{ZENODO_BASE}/{zip_filename}/content"

            # Download if not already on disk
            if not os.path.exists(zip_local_path):
                success = download_zip_stream(zip_url, zip_local_path, category)
                if not success:
                    print(f"  [SKIP] Could not download {zip_filename}")
                    continue
            else:
                size_mb = os.path.getsize(zip_local_path) / 1e6
                print(f"\n→ Using cached {zip_filename} ({size_mb:.1f} MB)")

            # Validate ZIP
            if not zipfile.is_zipfile(zip_local_path):
                print(f"  [ERROR] {zip_filename} is not a valid ZIP file. Deleting and skipping.")
                os.remove(zip_local_path)
                continue

            # Extract landmarks
            new_records = extract_landmarks_from_zip(
                zip_path=zip_local_path,
                output_dir=output_dir,
                extractor=extractor,
                max_per_class=max_per_class,
            )
            all_records.extend(new_records)

            # Delete ZIP to save disk space unless user wants to keep it
            if not keep_zips:
                os.remove(zip_local_path)
                print(f"  Deleted {zip_filename} to save space (landmark .npy files kept)")

    extractor.close()

    # Save updated labels CSV
    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset=["npy_path"])
    df.to_csv(labels_csv, index=False)

    unique_classes = sorted(df["label"].unique())
    print(f"\n{'='*65}")
    print(f"  REAL INCLUDE Dataset — Complete Summary")
    print(f"{'='*65}")
    print(f"  Total samples     : {len(df)}")
    print(f"  Unique ISL classes: {len(unique_classes)}")
    print(f"  Saved labels CSV  : {labels_csv}")
    print(f"{'='*65}")
    print(f"  Classes: {unique_classes}")
    print(f"{'='*65}\n")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download real INCLUDE ISL videos from Zenodo and extract landmarks."
    )
    parser.add_argument(
        "--categories", nargs="+", default=DEFAULT_CATEGORIES,
        choices=list(ZENODO_CATEGORY_ZIPS.keys()),
        help=f"Which INCLUDE categories to download. Default: {DEFAULT_CATEGORIES}"
    )
    parser.add_argument(
        "--max_per_class", type=int, default=None,
        help="Max videos per sign class (default: all available)"
    )
    parser.add_argument(
        "--keep_zips", action="store_true",
        help="Keep downloaded ZIP files after extraction (default: delete to save space)"
    )
    parser.add_argument(
        "--output_dir", default=LANDMARKS_DIR,
        help=f"Where to save .npy files. Default: {LANDMARKS_DIR}"
    )
    args = parser.parse_args()

    df = fetch_and_extract(
        categories=args.categories,
        output_dir=args.output_dir,
        labels_csv=LABELS_CSV,
        max_per_class=args.max_per_class,
        keep_zips=args.keep_zips,
    )

    print("Next step: pack the dataset splits and train:")
    print("  python src/preprocessing/run_pipeline.py --skip_download")
