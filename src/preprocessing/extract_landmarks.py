"""
MediaPipe Holistic Landmark Extractor for ISL Sign Videos.
Extracts 258 features per frame:
- Pose: 33 points x 4 (x, y, z, visibility) = 132
- Left Hand: 21 points x 3 (x, y, z) = 63
- Right Hand: 21 points x 3 (x, y, z) = 63
Total: 258 features per frame.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm
from typing import Tuple, Optional, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.config import (
    SEQ_LEN,
    TOTAL_FEATURES,
    POSE_FEATURES,
    LEFT_HAND_FEATURES,
    RIGHT_HAND_FEATURES,
)



class LandmarkExtractor:
    """MediaPipe Holistic wrapper for sign video feature extraction."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract_frame_landmarks(self, image_bgr: np.ndarray, results=None) -> np.ndarray:
        """
        Extract 258 feature vector from a single RGB image or pre-computed MediaPipe results.
        
        Returns:
            np.ndarray: Vector of shape (258,)
        """
        if results is None:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self.holistic.process(image_rgb)

        # Pose: 33 points * 4 (x, y, z, visibility)
        if results.pose_landmarks:
            pose = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
            ).flatten()
        else:
            pose = np.zeros(POSE_FEATURES)

        # Left Hand: 21 points * 3 (x, y, z)
        if results.left_hand_landmarks:
            lh = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
            ).flatten()
        else:
            lh = np.zeros(LEFT_HAND_FEATURES)

        # Right Hand: 21 points * 3 (x, y, z)
        if results.right_hand_landmarks:
            rh = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
            ).flatten()
        else:
            rh = np.zeros(RIGHT_HAND_FEATURES)

        feature_vector = np.concatenate([pose, lh, rh])
        assert feature_vector.shape[0] == TOTAL_FEATURES, f"Expected {TOTAL_FEATURES}, got {feature_vector.shape[0]}"
        return feature_vector

    def process_video(
        self, video_path: str, target_seq_len: int = SEQ_LEN
    ) -> np.ndarray:
        """
        Processes a video file and returns a landmark sequence of shape (target_seq_len, 258).
        """
        cap = cv2.VideoCapture(video_path)
        raw_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            feature_vector = self.extract_frame_landmarks(frame)
            raw_frames.append(feature_vector)

        cap.release()

        if len(raw_frames) == 0:
            # Empty video fallback
            return np.zeros((target_seq_len, TOTAL_FEATURES), dtype=np.float32)

        raw_sequence = np.array(raw_frames, dtype=np.float32)
        return self.normalize_sequence_length(raw_sequence, target_seq_len)

    @staticmethod
    def normalize_sequence_length(
        sequence: np.ndarray, target_seq_len: int = SEQ_LEN
    ) -> np.ndarray:
        """
        Resamples or pads sequence to exactly target_seq_len frames.
        """
        num_frames = sequence.shape[0]

        if num_frames == target_seq_len:
            return sequence
        elif num_frames > target_seq_len:
            # Uniform downsampling
            indices = np.linspace(0, num_frames - 1, target_seq_len, dtype=int)
            return sequence[indices]
        else:
            # Zero padding
            padding = np.zeros(
                (target_seq_len - num_frames, TOTAL_FEATURES), dtype=np.float32
            )
            return np.vstack([sequence, padding])

    def close(self):
        self.holistic.close()


def process_dataset_directory(
    input_dir: str, output_dir: str, csv_output_path: str, target_seq_len: int = SEQ_LEN
):
    """
    Processes all video files in input_dir (structured by class subfolders)
    and saves .npy landmark files to output_dir along with labels.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)

    extractor = LandmarkExtractor()
    records = []

    video_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                label = os.path.basename(root)
                video_files.append((os.path.join(root, file), label, file))

    print(f"Found {len(video_files)} video files in {input_dir}")

    for idx, (video_path, label, filename) in enumerate(tqdm(video_files, desc="Extracting landmarks")):
        video_id = os.path.splitext(filename)[0]
        try:
            landmarks = extractor.process_video(video_path, target_seq_len=target_seq_len)
            npy_filename = f"{label}_{video_id}_{idx}.npy"
            npy_save_path = os.path.join(output_dir, npy_filename)
            np.save(npy_save_path, landmarks)

            records.append({
                "video_id": video_id,
                "label": label,
                "npy_path": npy_save_path,
                "frames": landmarks.shape[0],
                "features": landmarks.shape[1],
            })
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

    extractor.close()

    df = pd.DataFrame(records)
    df.to_csv(csv_output_path, index=False)
    print(f"Saved {len(records)} landmark records to {csv_output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract MediaPipe Holistic landmarks from ISL gesture videos.")
    parser.add_argument("--input_dir", type=str, default="data/raw/sample", help="Input directory containing class subfolders")
    parser.add_argument("--output_dir", type=str, default="data/landmarks", help="Output directory for .npy landmark files")
    parser.add_argument("--csv_output", type=str, default="data/splits/labels.csv", help="CSV mapping output file")
    args = parser.parse_args()

    if os.path.exists(args.input_dir):
        process_dataset_directory(args.input_dir, args.output_dir, args.csv_output)
    else:
        print(f"Input directory {args.input_dir} does not exist yet.")
