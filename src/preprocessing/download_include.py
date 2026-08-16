"""
Fast Resilient Downloader for AI4Bharat INCLUDE Dataset from Zenodo API using curl.
Downloads gesture category zip files directly into data/raw/ and extracts video clips.
"""

import os
import sys
import zipfile
import subprocess
from typing import List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Verified Zenodo API file endpoints for INCLUDE gesture categories
ZENODO_FILES = {
    "Greetings_1": "https://zenodo.org/api/records/4010759/files/Greetings_1of2.zip/content",
    "Greetings_2": "https://zenodo.org/api/records/4010759/files/Greetings_2of2.zip/content",
    "Colours_1": "https://zenodo.org/api/records/4010759/files/Colours_1of2.zip/content",
    "Colours_2": "https://zenodo.org/api/records/4010759/files/Colours_2of2.zip/content",
    "Electronics_1": "https://zenodo.org/api/records/4010759/files/Electronics_1of2.zip/content",
    "Electronics_2": "https://zenodo.org/api/records/4010759/files/Electronics_2of2.zip/content",
    "Days_and_Time_1": "https://zenodo.org/api/records/4010759/files/Days_and_Time_1of3.zip/content",
    "Animals_1": "https://zenodo.org/api/records/4010759/files/Animals_1of2.zip/content",
}


def download_and_extract_category(category: str = "Greetings_1", output_dir: str = "data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    if category not in ZENODO_FILES:
        print(f"Unknown category '{category}'. Available: {list(ZENODO_FILES.keys())}")
        return

    url = ZENODO_FILES[category]
    zip_path = os.path.join(output_dir, f"{category}.zip")

    print(f"Downloading {category} from Zenodo...")
    cmd = [
        "curl",
        "-L",
        "-A", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        url,
        "-o", zip_path
    ]
    subprocess.run(cmd, check=True)

    print(f"Extracting {zip_path} into {output_dir}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(output_dir)
        print(f"Successfully extracted {category}!")
    except Exception as e:
        print(f"Error unzipping {zip_path}: {e}")

    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"Cleaned up zip archive {zip_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download and extract INCLUDE gesture dataset categories.")
    parser.add_argument("--categories", nargs="+", default=["Greetings_1"], help="Categories to download")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Target output directory")
    args = parser.parse_args()

    for cat in args.categories:
        download_and_extract_category(cat, args.output_dir)
