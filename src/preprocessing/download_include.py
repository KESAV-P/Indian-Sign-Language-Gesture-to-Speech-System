"""
INCLUDE ISL Dataset Downloader & Real Landmark Extractor.

Downloads the real AI4Bharat INCLUDE / INCLUDE-50 Indian Sign Language dataset
(263 word classes, ~4,287 real human signer videos, CC-BY 4.0) from HuggingFace
metadata + Zenodo video files, then extracts genuine MediaPipe Holistic 258-feature
landmark sequences from each video.

ALL SYNTHETIC DATA HAS BEEN REMOVED.
This pipeline only produces real landmark features extracted from real human signers.

Download strategy (tried in order):
  1. HuggingFace `datasets` library — streams INCLUDE video metadata and downloads
     raw video bytes from Zenodo per-sample.
  2. Official AI4Bharat bash download script — bulk-downloads all videos from Zenodo
     into data/raw/include/ and processes them locally.

Usage:
    python src/preprocessing/download_include.py               # Full 263-class INCLUDE
    python src/preprocessing/download_include.py --subset 50   # INCLUDE-50 subset only
    python src/preprocessing/download_include.py --max_samples 500  # Quick test run
"""

import os
import sys
import re
import json
import subprocess
import tempfile
import shutil
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.config import SEQ_LEN, TOTAL_FEATURES
from src.preprocessing.extract_landmarks import LandmarkExtractor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INCLUDE_HF_DATASET = "ai4bharat/INCLUDE"
INCLUDE_GITHUB_REPO = "https://github.com/AI4Bharat/INCLUDE"
ZENODO_DOWNLOAD_SCRIPT = "https://raw.githubusercontent.com/AI4Bharat/INCLUDE/master/data/download.sh"

RAW_VIDEO_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "include")
LANDMARKS_DIR = os.path.join(PROJECT_ROOT, "data", "landmarks")
LABELS_CSV    = os.path.join(PROJECT_ROOT, "data", "splits", "labels.csv")

# Folder names that are CATEGORY buckets, not actual ISL sign labels.
# Videos inside these folders must have their label derived from the filename
# prefix instead.  Add any future category-level folder names here.
CATEGORY_FOLDER_NAMES: set = {
    "greetings",
    "include",
    "days_and_time",
    "days and time",
    "electronics",
    "food",
    "numbers",
    "colors",
    "colours",
    "family",
    "animals",
    "body",
    "actions",
    "raw",          # safety: never label from a top-level raw/ folder
}

# Pattern that matches a video-camera ID suffix at the END of a filename stem,
# e.g. "_MVI_0043", "_0043", "_ (2)", "_ (1)", "_MVI_9914"
# Anything after this pattern (and the pattern itself) is stripped.
_VIDEO_ID_SUFFIX_RE = re.compile(
    r"[_\s]+(?:MVI[_\s]+)?[0-9]+(?:[_\s]*\([0-9]+\))?$",
    re.IGNORECASE,
)


def clean_label(label: str) -> str:
    """Clean label: strip leading numeric indices, title-case."""
    cleaned = re.sub(r"^\d+[\.\-\_\s]+", "", label).strip()
    return cleaned.title() if cleaned else label.strip()


def _label_from_filename(fname: str) -> Optional[str]:
    """
    Try to extract an ISL word label from a video filename.

    Handles two filename conventions found in the INCLUDE dataset:

    1. Word-prefixed  : "Alright_MVI_0043.MOV"  -> "Alright"
                        "Good_Morning_MVI_0042.MOV" -> "Good Morning"
                        "Good_afternoon_MVI_0050_ (2).MOV" -> "Good Afternoon"
    2. Bare camera IDs: "MVI_0029.MOV"           -> None  (fall back to folder)

    Steps:
      a) Strip file extension.
      b) Strip the trailing camera/sequence ID (_MVI_NNNN or just _NNNN).
      c) If nothing is left (bare MVI_* filenames), return None.
      d) Replace underscores with spaces, title-case, and return.
    """
    stem = os.path.splitext(fname)[0]          # remove extension
    # Strip trailing video-ID suffix
    candidate = _VIDEO_ID_SUFFIX_RE.sub("", stem).strip()
    # If the candidate still starts with "MVI" it's a bare camera filename
    if not candidate or re.match(r"^MVI\b", candidate, re.IGNORECASE):
        return None
    # Replace underscores/hyphens with spaces, clean numeric prefix, title-case
    candidate = candidate.replace("_", " ").replace("-", " ")
    candidate = re.sub(r"^\d+[.\s]+", "", candidate).strip()
    return candidate.title() if candidate else None


def _is_category_folder(folder_name: str) -> bool:
    """
    Return True if `folder_name` is a category-level bucket name rather than
    an actual ISL sign word.  Matching is case-insensitive.
    """
    return folder_name.strip().lower() in CATEGORY_FOLDER_NAMES


# ---------------------------------------------------------------------------
# Path A: HuggingFace datasets library (recommended — no raw video storage)
# ---------------------------------------------------------------------------

def download_via_huggingface(
    output_landmarks_dir: str = LANDMARKS_DIR,
    include_50_only: bool = False,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Download INCLUDE dataset via HuggingFace `datasets` library.

    The HuggingFace INCLUDE dataset contains metadata (label, video_path, include_50 flag)
    for all ~4,287 videos. We use this metadata to construct Zenodo download URLs and
    fetch each video, then run LandmarkExtractor on it.

    Args:
        output_landmarks_dir: Where to save .npy landmark files.
        include_50_only: If True, only process the INCLUDE-50 subset (50 classes, ~820 videos).
        max_samples: If set, only process this many samples (useful for quick testing).

    Returns:
        List of record dicts with video_id, label, npy_path, frames, features.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "HuggingFace `datasets` library not installed.\n"
            "Run: pip install datasets\n"
            "Then retry."
        )

    print(f"\n{'='*60}")
    print(f"  Downloading AI4Bharat INCLUDE ISL Dataset")
    print(f"  Source: HuggingFace ({INCLUDE_HF_DATASET})")
    print(f"  Mode: {'INCLUDE-50 (50 classes)' if include_50_only else 'Full INCLUDE (263 classes)'}")
    print(f"{'='*60}\n")

    print("Loading INCLUDE metadata from HuggingFace...")
    try:
        dataset = load_dataset(INCLUDE_HF_DATASET, trust_remote_code=True)
    except Exception as e:
        print(f"[ERROR] Could not load HuggingFace dataset: {e}")
        print("Falling back to Zenodo bulk download method...")
        return download_via_zenodo_script(
            output_landmarks_dir=output_landmarks_dir,
            include_50_only=include_50_only,
            max_samples=max_samples,
        )

    os.makedirs(output_landmarks_dir, exist_ok=True)
    extractor = LandmarkExtractor()
    records = []
    processed = 0
    failed = 0

    for split_name in ["train", "val", "test"]:
        if split_name not in dataset:
            continue
        split_data = dataset[split_name]

        for sample in split_data:
            if max_samples is not None and processed >= max_samples:
                break

            # Filter to INCLUDE-50 subset if requested
            if include_50_only and not sample.get("include_50", False):
                continue

            label = clean_label(sample.get("label", ""))
            video_path_meta = sample.get("video_path", "")
            if not label or not video_path_meta:
                continue

            video_id = Path(video_path_meta).stem
            safe_label = label.replace(" ", "_")
            npy_filename = f"include_{safe_label}_{video_id}.npy"
            npy_save_path = os.path.join(output_landmarks_dir, npy_filename)

            # Skip if already extracted
            if os.path.exists(npy_save_path):
                records.append({
                    "video_id": video_id,
                    "label": label,
                    "npy_path": npy_save_path,
                    "frames": SEQ_LEN,
                    "features": TOTAL_FEATURES,
                    "source": "include_hf",
                    "split": split_name,
                })
                processed += 1
                continue

            # Try to get video bytes from the sample (HF may embed them)
            video_bytes = sample.get("video", None)
            if video_bytes is None:
                # No inline video — try downloading from Zenodo by constructing the URL
                video_bytes = _fetch_zenodo_video(video_path_meta)

            if video_bytes is None:
                print(f"  [SKIP] Could not fetch video for {video_id} ({label})")
                failed += 1
                continue

            # Write to temp file, process, delete
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    if isinstance(video_bytes, bytes):
                        tmp.write(video_bytes)
                    elif hasattr(video_bytes, "read"):
                        tmp.write(video_bytes.read())
                    else:
                        tmp.write(bytes(video_bytes))
                    tmp_path = tmp.name

                landmarks = extractor.process_video(tmp_path, target_seq_len=SEQ_LEN)
                os.unlink(tmp_path)

                np.save(npy_save_path, landmarks)
                records.append({
                    "video_id": video_id,
                    "label": label,
                    "npy_path": npy_save_path,
                    "frames": SEQ_LEN,
                    "features": TOTAL_FEATURES,
                    "source": "include_hf",
                    "split": split_name,
                })
                processed += 1

                if processed % 50 == 0:
                    print(f"  Progress: {processed} videos processed, {failed} failed...")

            except Exception as e:
                print(f"  [ERROR] Processing {video_id}: {e}")
                failed += 1
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        if max_samples is not None and processed >= max_samples:
            break

    extractor.close()
    print(f"\n✓ Processed {processed} real ISL videos | {failed} failed")
    return records


def _fetch_zenodo_video(video_path_meta: str) -> Optional[bytes]:
    """
    Attempt to download a video from the AI4Bharat Zenodo archive.
    The INCLUDE GitHub repo documents the Zenodo record base URL.
    """
    # The Zenodo record for INCLUDE is at: https://zenodo.org/record/4010759
    # Videos are stored as: <category>/<word>/<filename>.mp4
    ZENODO_BASE = "https://zenodo.org/record/4010759/files"
    filename = Path(video_path_meta).name
    url = f"{ZENODO_BASE}/{filename}"

    try:
        resp = requests.get(url, timeout=30, stream=True)
        if resp.status_code == 200:
            return resp.content
    except Exception:
        pass

    # Try alternate path structure
    try:
        parts = Path(video_path_meta).parts
        if len(parts) >= 2:
            url2 = f"{ZENODO_BASE}/{parts[-2]}/{parts[-1]}"
            resp2 = requests.get(url2, timeout=30, stream=True)
            if resp2.status_code == 200:
                return resp2.content
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Path B: Zenodo bulk download via official bash script
# ---------------------------------------------------------------------------

def download_via_zenodo_script(
    output_landmarks_dir: str = LANDMARKS_DIR,
    include_50_only: bool = False,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Download the full INCLUDE dataset from Zenodo using the official AI4Bharat
    download script, then extract landmarks from all downloaded videos.

    This downloads the raw videos (~15-20 GB for full dataset, ~3 GB for INCLUDE-50)
    to data/raw/include/, then runs LandmarkExtractor on each .mp4 file.
    """
    print(f"\n{'='*60}")
    print(f"  Path B: Zenodo Bulk Download (AI4Bharat INCLUDE)")
    print(f"  This will download raw ISL video files to: {RAW_VIDEO_DIR}")
    print(f"  Estimated size: ~3 GB (INCLUDE-50) | ~15 GB (full)")
    print(f"{'='*60}\n")

    os.makedirs(RAW_VIDEO_DIR, exist_ok=True)

    # Download the official AI4Bharat download script
    print("Fetching official AI4Bharat download script from GitHub...")
    script_path = os.path.join(RAW_VIDEO_DIR, "download.sh")

    try:
        resp = requests.get(ZENODO_DOWNLOAD_SCRIPT, timeout=30)
        resp.raise_for_status()
        with open(script_path, "w") as f:
            f.write(resp.text)
        os.chmod(script_path, 0o755)
        print(f"✓ Download script saved to: {script_path}")
    except Exception as e:
        print(f"[ERROR] Could not fetch download script: {e}")
        print(f"Please manually download the script from:")
        print(f"  {INCLUDE_GITHUB_REPO}/blob/master/data/download.sh")
        print(f"and run it in: {RAW_VIDEO_DIR}")
        return []

    # Run the download script
    print(f"\nRunning download script (this may take a long time)...")
    print(f"  Downloading to: {RAW_VIDEO_DIR}")

    try:
        result = subprocess.run(
            ["bash", script_path],
            cwd=RAW_VIDEO_DIR,
            timeout=7200,  # 2 hour timeout
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"[WARN] Download script exited with code {result.returncode}")
    except subprocess.TimeoutExpired:
        print("[WARN] Download timed out after 2 hours. Processing what was downloaded...")
    except Exception as e:
        print(f"[ERROR] Could not run download script: {e}")
        return []

    # Now extract landmarks from all downloaded videos
    print(f"\nExtracting landmarks from downloaded videos in {RAW_VIDEO_DIR}...")
    records = _extract_landmarks_from_directory(
        raw_dir=RAW_VIDEO_DIR,
        output_dir=output_landmarks_dir,
        include_50_only=include_50_only,
        max_samples=max_samples,
    )
    return records


# ---------------------------------------------------------------------------
# Local video directory processing
# ---------------------------------------------------------------------------

def process_local_video_directory(
    raw_dir: str,
    output_landmarks_dir: str = LANDMARKS_DIR,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Process any local directory of videos organized as:
        raw_dir/<class_name>/<video_file>.mp4

    Extracts real MediaPipe landmarks from each video.
    This handles the Greetings dataset, INCLUDE downloaded locally, or any custom ISL videos.
    """
    if not os.path.exists(raw_dir):
        print(f"[INFO] Directory {raw_dir} does not exist. Skipping local processing.")
        return []

    return _extract_landmarks_from_directory(
        raw_dir=raw_dir,
        output_dir=output_landmarks_dir,
        max_samples=max_samples,
    )


def _extract_landmarks_from_directory(
    raw_dir: str,
    output_dir: str,
    include_50_only: bool = False,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Walk a directory of videos and extract MediaPipe landmarks.

    Label resolution strategy (in priority order):
      1. If the immediate parent folder name is a KNOWN CATEGORY BUCKET
         (e.g. "Greetings", "Electronics") — derive the label from the
         FILENAME PREFIX using _label_from_filename().
      2. If _label_from_filename() returns None for that file (bare MVI_*
         filenames) — fall back to the GRANDPARENT folder name, which is
         typically the word-level subfolder (e.g. "48. Hello").
      3. Otherwise — use the cleaned parent folder name directly.

    This guarantees that category-level folder names ("Greetings", etc.)
    never become class labels in the output CSV.

    Returns list of record dicts.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect all video files with resolved labels
    video_files: List[tuple] = []
    category_label_warnings: set = set()

    for root, _, files in os.walk(raw_dir):
        parent_folder = os.path.basename(root)
        grandparent_folder = os.path.basename(os.path.dirname(root))

        for fname in files:
            if not fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue

            full_path = os.path.join(root, fname)

            if _is_category_folder(parent_folder):
                # ── Strategy 1: derive label from filename prefix ─────────
                label_from_fn = _label_from_filename(fname)
                if label_from_fn:
                    label = label_from_fn
                else:
                    # ── Strategy 2: fall back to grandparent folder name ──
                    label = clean_label(grandparent_folder)
                    if _is_category_folder(grandparent_folder):
                        # Both parent and grandparent are category folders —
                        # skip this file, we cannot determine its label.
                        print(
                            f"  [SKIP] Cannot determine label for {full_path} "
                            f"(both parent '{parent_folder}' and grandparent "
                            f"'{grandparent_folder}' are category folders)"
                        )
                        continue
                if parent_folder not in category_label_warnings:
                    print(
                        f"  [INFO] Folder '{parent_folder}' is a category bucket — "
                        f"labels derived from filename prefixes."
                    )
                    category_label_warnings.add(parent_folder)
            else:
                # ── Strategy 3: use the parent folder name directly ───────
                label = clean_label(parent_folder)

            # Final guard: if after all resolution the label is still a
            # category name, refuse to add it.
            if _is_category_folder(label.lower()):
                print(
                    f"  [SKIP] Resolved label '{label}' for {fname} is still a "
                    f"category-level name — skipping to avoid mislabeling."
                )
                continue

            video_files.append((full_path, label, fname))

    if not video_files:
        print(f"[WARN] No video files found in {raw_dir}")
        return []

    if max_samples:
        video_files = video_files[:max_samples]

    print(f"Found {len(video_files)} real human ISL video files in {raw_dir}")
    print(f"Extracting MediaPipe landmarks (258 features/frame, {SEQ_LEN} frames/video)...")

    # Show label distribution before extraction
    from collections import Counter
    label_dist = Counter(lbl for _, lbl, _ in video_files)
    print(f"Label distribution: {dict(sorted(label_dist.items()))}")

    extractor = LandmarkExtractor()
    records = []
    failed = 0

    for idx, (video_path, label, fname) in enumerate(video_files):
        video_id = os.path.splitext(fname)[0]
        safe_label = label.replace(" ", "_")
        npy_filename = f"{safe_label}_{video_id}_{idx:05d}.npy"
        npy_save_path = os.path.join(output_dir, npy_filename)

        if os.path.exists(npy_save_path):
            records.append({
                "video_id": video_id,
                "label": label,
                "npy_path": npy_save_path,
                "frames": SEQ_LEN,
                "features": TOTAL_FEATURES,
                "source": "real",
            })
            continue

        try:
            landmarks = extractor.process_video(video_path, target_seq_len=SEQ_LEN)
            np.save(npy_save_path, landmarks)
            records.append({
                "video_id": video_id,
                "label": label,
                "npy_path": npy_save_path,
                "frames": SEQ_LEN,
                "features": TOTAL_FEATURES,
                "source": "real",
            })

            if (idx + 1) % 20 == 0:
                print(f"  [{idx+1}/{len(video_files)}] Processed {label}/{fname}")

        except Exception as e:
            print(f"  [ERROR] {video_path}: {e}")
            failed += 1

    extractor.close()
    print(f"\n✓ Landmark extraction complete: {len(records)} successful, {failed} failed")
    return records


# ---------------------------------------------------------------------------
# Master build function
# ---------------------------------------------------------------------------

def build_real_dataset(
    output_landmarks_dir: str = LANDMARKS_DIR,
    labels_csv: str = LABELS_CSV,
    include_50_only: bool = True,
    max_samples: Optional[int] = None,
    force_redownload: bool = False,
) -> pd.DataFrame:
    """
    Builds a fully REAL ISL landmark dataset.

    Pipeline:
    1. Check for any pre-existing local videos in data/raw/include/
    2. If found, extract landmarks from those
    3. Also check data/raw/Greetings/ for local greetings data
    4. If not enough data, download from HuggingFace / Zenodo
    5. Combine all sources into labels.csv

    Args:
        output_landmarks_dir: Where to save .npy files.
        labels_csv: Path to output CSV with all sample metadata.
        include_50_only: If True, use only the 50-class subset (faster).
        max_samples: Cap total samples (for testing).
        force_redownload: Re-download even if landmarks already exist.

    Returns:
        DataFrame with all extracted samples.
    """
    os.makedirs(output_landmarks_dir, exist_ok=True)
    os.makedirs(os.path.dirname(labels_csv), exist_ok=True)

    all_records: List[Dict] = []

    # ------------------------------------------------------------------
    # Check for existing .npy files — skip re-extraction if possible
    # ------------------------------------------------------------------
    existing_npy = list(Path(output_landmarks_dir).glob("*.npy"))
    if existing_npy and not force_redownload:
        print(f"\n✓ Found {len(existing_npy)} existing landmark .npy files in {output_landmarks_dir}")
        print("  (Use --force_redownload to re-extract from scratch)")

        # Try to load existing labels CSV — but validate it before returning
        if os.path.exists(labels_csv):
            df = pd.read_csv(labels_csv)
            # Guard: refuse to silently return a stale CSV with bad source tags
            # or category-level labels.  Let caller decide what to do.
            non_real = df[~df["source"].isin(["real", "include_hf"])] if "source" in df.columns else pd.DataFrame()
            category_rows = df[df["label"].str.lower().isin(CATEGORY_FOLDER_NAMES)] if "label" in df.columns else pd.DataFrame()
            if not non_real.empty or not category_rows.empty:
                print(
                    f"  [WARN] Existing labels.csv has {len(non_real)} non-real rows "
                    f"and {len(category_rows)} category-labeled rows.\n"
                    f"  Re-running extraction to fix stale labels.\n"
                    f"  (Pass --force_redownload to also re-extract existing .npy files.)"
                )
                # Don't return stale CSV — fall through to re-extraction
            else:
                print(f"  Loaded {len(df)} records from existing {labels_csv}")
                print(f"  Classes: {sorted(df['label'].unique())}")
                return df
        # else: no CSV yet — fall through to build from scratch

    # ------------------------------------------------------------------
    # Step 1: Process any local Greetings videos
    # ------------------------------------------------------------------
    greetings_dir = os.path.join(PROJECT_ROOT, "data", "raw", "Greetings")
    if os.path.exists(greetings_dir) and not force_redownload:
        print(f"\n→ Processing local Greetings videos from {greetings_dir}...")
        greetings_records = process_local_video_directory(
            raw_dir=greetings_dir,
            output_landmarks_dir=output_landmarks_dir,
            max_samples=max_samples,
        )
        all_records.extend(greetings_records)
        print(f"  Added {len(greetings_records)} real Greetings samples")

    # ------------------------------------------------------------------
    # Step 2: Process any pre-downloaded INCLUDE videos
    # ------------------------------------------------------------------
    if os.path.exists(RAW_VIDEO_DIR) and any(Path(RAW_VIDEO_DIR).rglob("*.mp4")):
        n_local = len(list(Path(RAW_VIDEO_DIR).rglob("*.mp4")))
        print(f"\n→ Found {n_local} pre-downloaded INCLUDE video files in {RAW_VIDEO_DIR}")
        local_records = _extract_landmarks_from_directory(
            raw_dir=RAW_VIDEO_DIR,
            output_dir=output_landmarks_dir,
            include_50_only=include_50_only,
            max_samples=max_samples,
        )
        all_records.extend(local_records)
        print(f"  Added {len(local_records)} INCLUDE landmark samples")

    # ------------------------------------------------------------------
    # Step 3: If we still have no data, download from HuggingFace
    # ------------------------------------------------------------------
    if not all_records:
        print(f"\n→ No local data found. Downloading from HuggingFace ({INCLUDE_HF_DATASET})...")
        hf_records = download_via_huggingface(
            output_landmarks_dir=output_landmarks_dir,
            include_50_only=include_50_only,
            max_samples=max_samples,
        )
        all_records.extend(hf_records)

    if not all_records:
        raise RuntimeError(
            "\n[FATAL] No real ISL data could be downloaded or found.\n"
            "Please try one of the following:\n"
            "  1. Place real ISL videos in data/raw/include/<class_name>/<video.mp4>\n"
            "  2. Run with internet access to download from HuggingFace\n"
            "  3. Download the INCLUDE dataset from:\n"
            "       https://github.com/AI4Bharat/INCLUDE\n"
            "       https://zenodo.org/record/4010759\n"
        )

    # ------------------------------------------------------------------
    # Save unified labels CSV
    # ------------------------------------------------------------------
    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset=["npy_path"])
    df.to_csv(labels_csv, index=False)

    unique_classes = sorted(df["label"].unique())
    print(f"\n{'='*60}")
    print(f"  REAL Dataset Summary")
    print(f"{'='*60}")
    print(f"  Total samples : {len(df)}")
    print(f"  Unique classes: {len(unique_classes)}")
    print(f"  Classes       : {unique_classes}")
    print(f"  Saved CSV to  : {labels_csv}")
    print(f"{'='*60}\n")

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and extract real AI4Bharat INCLUDE ISL dataset."
    )
    parser.add_argument(
        "--subset", type=int, choices=[50, 263], default=50,
        help="Use INCLUDE-50 (50 classes) or full INCLUDE (263 classes). Default: 50"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit total samples processed (for testing). Default: no limit"
    )
    parser.add_argument(
        "--force_redownload", action="store_true",
        help="Re-download and re-extract even if .npy files already exist"
    )
    parser.add_argument(
        "--landmarks_dir", type=str, default=LANDMARKS_DIR,
        help=f"Output directory for landmark .npy files. Default: {LANDMARKS_DIR}"
    )
    parser.add_argument(
        "--labels_csv", type=str, default=LABELS_CSV,
        help=f"Output CSV path. Default: {LABELS_CSV}"
    )
    args = parser.parse_args()

    df = build_real_dataset(
        output_landmarks_dir=args.landmarks_dir,
        labels_csv=args.labels_csv,
        include_50_only=(args.subset == 50),
        max_samples=args.max_samples,
        force_redownload=args.force_redownload,
    )

    print(f"\nDataset ready. Next step: run pack_and_split_dataset() in build_dataset.py")
    print(f"  or run:  python src/preprocessing/run_pipeline.py")
