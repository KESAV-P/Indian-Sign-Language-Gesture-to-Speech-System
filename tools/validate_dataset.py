"""
tools/validate_dataset.py — ISL-Speak Dataset Integrity Validator
==================================================================

Runs four checks on data/splits/labels.csv and the referenced .npy files:

  1. Per-class sample counts  — flags any class with < MIN_SAMPLES samples.
  2. Source column audit       — hard error if ANY row has source != "real" or
                                "include_hf", unless --allow-synthetic is passed.
  3. Duplicate / near-duplicate detection
                               — hashes each .npy (and a rounded version) to catch
                                 byte-identical or near-identical arrays across
                                 different labels (cosine sim > 0.999).
  4. Category-label guard      — re-checks that no label is a known category/folder
                                 name (e.g. "Greetings", "Electronics") rather than
                                 an actual ISL sign word.

Usage:
    python tools/validate_dataset.py                      # strict: no synthetic
    python tools/validate_dataset.py --allow-synthetic    # relax source check
    python tools/validate_dataset.py --labels data/splits/labels.csv
    python tools/validate_dataset.py --min-samples 10
"""

import argparse
import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Constants (keep in sync with download_include.py)
# ---------------------------------------------------------------------------
MIN_SAMPLES_DEFAULT = 15

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
    "raw",
}

REAL_SOURCES = {"real", "include_hf"}

COSINE_SIM_THRESHOLD = 0.999   # flag pairs with cosine similarity above this

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(arr: np.ndarray) -> str:
    """SHA-256 of raw bytes of a float32 array (exact duplicate detector)."""
    return hashlib.sha256(arr.astype(np.float32).tobytes()).hexdigest()


def _sha256_rounded(arr: np.ndarray, decimals: int = 3) -> str:
    """SHA-256 of array rounded to `decimals` places (near-duplicate detector)."""
    return hashlib.sha256(np.round(arr.astype(np.float32), decimals).tobytes()).hexdigest()


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flattened arrays."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def _print_section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def _print_ok(msg: str):
    print(f"  ✅  {msg}")


def _print_warn(msg: str):
    print(f"  ⚠️   {msg}")


def _print_fail(msg: str):
    print(f"  ❌  {msg}")


# ---------------------------------------------------------------------------
# Check 1: Per-class sample counts
# ---------------------------------------------------------------------------

def check_class_counts(df: pd.DataFrame, min_samples: int) -> List[str]:
    """Return list of error strings for classes with < min_samples samples."""
    errors = []
    counts = df["label"].value_counts().sort_index()
    print(f"\n  Per-class sample counts (minimum required: {min_samples}):\n")
    print(f"  {'Label':<30} {'Samples':>8}  {'Status'}")
    print(f"  {'-'*30} {'-'*8}  {'-'*10}")
    for label, count in counts.items():
        status = "OK" if count >= min_samples else f"BELOW MIN ({min_samples})"
        flag = "✅" if count >= min_samples else "❌"
        print(f"  {flag} {label:<30} {count:>8}  {status}")
        if count < min_samples:
            errors.append(f"Class '{label}' has only {count} samples (< {min_samples})")
    print(f"\n  Total classes : {len(counts)}")
    print(f"  Total samples : {len(df)}")
    return errors


# ---------------------------------------------------------------------------
# Check 2: Source column audit
# ---------------------------------------------------------------------------

def check_sources(df: pd.DataFrame, allow_synthetic: bool) -> List[str]:
    """Return list of error strings for non-real source rows."""
    errors = []
    if "source" not in df.columns:
        _print_warn("'source' column missing from labels.csv — cannot verify.")
        return errors

    source_counts = df["source"].value_counts()
    print(f"\n  Source breakdown:")
    for src, cnt in source_counts.items():
        is_real = src in REAL_SOURCES
        flag = "✅" if is_real else ("⚠️ " if allow_synthetic else "❌")
        print(f"    {flag} {src:<25} {cnt:>6} rows")

    non_real = df[~df["source"].isin(REAL_SOURCES)]
    if not non_real.empty:
        msg = (
            f"{len(non_real)} rows have non-real source: "
            f"{sorted(non_real['source'].unique())}"
        )
        if allow_synthetic:
            _print_warn(msg + " (--allow-synthetic: treated as warning)")
        else:
            _print_fail(msg)
            errors.append(msg)
    else:
        _print_ok("100% source == real")

    return errors


# ---------------------------------------------------------------------------
# Check 3: Duplicate / near-duplicate detection
# ---------------------------------------------------------------------------

def check_duplicates(df: pd.DataFrame) -> List[str]:
    """
    Hash every .npy array (exact + rounded) and flag cross-label collisions.
    Also checks high-cosine-similarity pairs across different labels.
    Returns list of error strings.
    """
    errors = []

    # Maps: hash -> list of (label, npy_path)
    exact_map: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    rounded_map: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    loaded: List[Tuple[str, str, np.ndarray]] = []   # (label, path, array)
    missing = 0

    print(f"\n  Loading {len(df)} .npy files for duplicate check...")

    for _, row in df.iterrows():
        npy_path = row["npy_path"]
        label = row["label"]
        if not os.path.exists(npy_path):
            missing += 1
            continue
        try:
            arr = np.load(npy_path).astype(np.float32)
        except Exception as e:
            _print_warn(f"Could not load {npy_path}: {e}")
            continue

        exact_map[_sha256(arr)].append((label, npy_path))
        rounded_map[_sha256_rounded(arr)].append((label, npy_path))
        loaded.append((label, npy_path, arr))

    if missing:
        _print_warn(f"{missing} .npy files referenced in CSV are missing from disk.")

    # --- Exact duplicates across different labels ---
    exact_cross = 0
    for h, entries in exact_map.items():
        labels_seen = {e[0] for e in entries}
        if len(labels_seen) > 1:
            paths = [e[1] for e in entries]
            msg = (
                f"EXACT duplicate array shared across labels {sorted(labels_seen)}: "
                f"{[os.path.basename(p) for p in paths[:4]]}"
            )
            _print_fail(msg)
            errors.append(msg)
            exact_cross += 1

    if exact_cross == 0:
        _print_ok("No exact cross-label duplicates found.")

    # --- Near-duplicates (rounded hash) across different labels ---
    near_cross = 0
    for h, entries in rounded_map.items():
        labels_seen = {e[0] for e in entries}
        if len(labels_seen) > 1:
            # Confirm with cosine similarity
            arrs = [(e[0], e[1]) for e in entries]
            msg = (
                f"NEAR-DUPLICATE (rounded hash) across labels {sorted(labels_seen)}: "
                f"{[os.path.basename(e[1]) for e in entries[:4]]}"
            )
            _print_fail(msg)
            errors.append(msg)
            near_cross += 1

    if near_cross == 0:
        _print_ok("No near-duplicate (rounded) cross-label arrays found.")

    # --- Cosine similarity scan (O(n²) — only if dataset is small enough) ---
    if len(loaded) <= 600:
        print(f"  Running cosine similarity scan on {len(loaded)} arrays...")
        cosine_errors = 0
        for i in range(len(loaded)):
            label_i, path_i, arr_i = loaded[i]
            for j in range(i + 1, len(loaded)):
                label_j, path_j, arr_j = loaded[j]
                if label_i == label_j:
                    continue   # same-class duplicates are OK
                sim = _cosine_sim(arr_i, arr_j)
                if sim > COSINE_SIM_THRESHOLD:
                    msg = (
                        f"HIGH cosine similarity ({sim:.5f}) between different labels: "
                        f"'{label_i}' ({os.path.basename(path_i)}) vs "
                        f"'{label_j}' ({os.path.basename(path_j)})"
                    )
                    _print_fail(msg)
                    errors.append(msg)
                    cosine_errors += 1
        if cosine_errors == 0:
            _print_ok(f"Cosine similarity scan passed (threshold: {COSINE_SIM_THRESHOLD}).")
    else:
        _print_warn(
            f"Dataset has {len(loaded)} samples — skipping O(n²) cosine scan "
            f"(would be too slow). Use exact/rounded hash checks above instead."
        )

    return errors


# ---------------------------------------------------------------------------
# Check 4: Category-label guard
# ---------------------------------------------------------------------------

def check_category_labels(df: pd.DataFrame) -> List[str]:
    """Return errors for any label that is a category/folder-level name."""
    errors = []
    bad_labels = [
        lbl for lbl in df["label"].unique()
        if lbl.strip().lower() in CATEGORY_FOLDER_NAMES
    ]
    if bad_labels:
        for lbl in sorted(bad_labels):
            count = (df["label"] == lbl).sum()
            msg = (
                f"Label '{lbl}' is a CATEGORY FOLDER NAME, not a specific sign "
                f"({count} rows affected) — this is the Greetings mislabeling bug."
            )
            _print_fail(msg)
            errors.append(msg)
    else:
        _print_ok("No category-level folder names used as class labels.")
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ISL-Speak Dataset Integrity Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--labels",
        default=os.path.join(PROJECT_ROOT, "data", "splits", "labels.csv"),
        help="Path to labels.csv (default: data/splits/labels.csv)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=MIN_SAMPLES_DEFAULT,
        dest="min_samples",
        help=f"Minimum samples per class (default: {MIN_SAMPLES_DEFAULT})",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        dest="allow_synthetic",
        help="Downgrade non-real source rows from ERROR to WARNING",
    )
    args = parser.parse_args()

    print(f"\n{'#'*65}")
    print(f"  ISL-Speak Dataset Integrity Validator")
    print(f"{'#'*65}")
    print(f"  Labels CSV : {args.labels}")
    print(f"  Min samples: {args.min_samples}")
    print(f"  Allow synth: {args.allow_synthetic}")

    if not os.path.exists(args.labels):
        print(f"\n❌ FATAL: labels.csv not found at {args.labels}")
        print("   Run: python src/preprocessing/run_pipeline.py --skip_download")
        sys.exit(1)

    df = pd.read_csv(args.labels)
    print(f"  Loaded     : {len(df)} rows")

    all_errors: List[str] = []

    # ── Check 1: per-class counts ────────────────────────────────────────
    _print_section("CHECK 1 — Per-class sample counts")
    all_errors.extend(check_class_counts(df, args.min_samples))

    # ── Check 2: source audit ────────────────────────────────────────────
    _print_section("CHECK 2 — Source column audit")
    all_errors.extend(check_sources(df, args.allow_synthetic))

    # ── Check 3: duplicate detection ─────────────────────────────────────
    _print_section("CHECK 3 — Duplicate / near-duplicate detection")
    all_errors.extend(check_duplicates(df))

    # ── Check 4: category-label guard ────────────────────────────────────
    _print_section("CHECK 4 — Category-label guard")
    all_errors.extend(check_category_labels(df))

    # ── Summary ──────────────────────────────────────────────────────────
    _print_section("SUMMARY")
    if all_errors:
        print(f"\n  ❌ {len(all_errors)} ERRORS found:\n")
        for i, err in enumerate(all_errors, 1):
            print(f"    {i}. {err}")
        print(f"\n  Fix the errors above before retraining.\n")
        sys.exit(1)
    else:
        print(f"\n  ✅ All checks passed — dataset is clean and ready for training.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
