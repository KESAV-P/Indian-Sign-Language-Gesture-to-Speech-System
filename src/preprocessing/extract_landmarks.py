"""
MediaPipe Landmark Extractor for ISL Sign Videos — CPU-only TFLite backend.
Extracts 258 features per frame:
- Pose: 33 points x 4 (x, y, z, visibility) = 132
- Left Hand: 21 points x 3 (x, y, z) = 63
- Right Hand: 21 points x 3 (x, y, z) = 63
Total: 258 features per frame.

Uses ai_edge_litert (Google's standalone TFLite runtime) to run MediaPipe
hand_landmark_full.tflite and pose_landmark_full.tflite directly on CPU,
bypassing the macOS Metal crash in mediapipe 1.0.x.

Landmark detection is hand-tracked: we use pose keypoints to locate the wrists,
then crop/resize hand regions for the hand landmark model. This is MediaPipe's
own two-stage approach, implemented here without Metal dependencies.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Optional, List, Dict

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

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "mediapipe_models")
_POSE_DETECT_MODEL = os.path.join(_MODELS_DIR, "pose_detection.tflite")
_POSE_LANDMARK_MODEL = os.path.join(_MODELS_DIR, "pose_landmark_full.tflite")
_HAND_LANDMARK_MODEL = os.path.join(_MODELS_DIR, "hand_landmark_full.tflite")
_PALM_DETECT_MODEL  = os.path.join(_MODELS_DIR, "palm_detection_full.onnx")  # saved as tflite

# Pose landmark indices for left/right wrist (used to locate hands)
_LEFT_WRIST_IDX  = 15
_RIGHT_WRIST_IDX = 16
_LEFT_ELBOW_IDX  = 13
_RIGHT_ELBOW_IDX = 14

# Hand crop scale factor relative to elbow→wrist distance
_HAND_CROP_SCALE  = 2.5   # generous crop so resting hands still fit
_HAND_MODEL_SIZE  = 224   # hand_landmark_full input: 224x224
_POSE_DETECT_SIZE = 224
_POSE_LANDMARK_SIZE = 256
_PALM_DETECT_SIZE = 192   # palm_detection_full input: 192x192
_PALM_SCORE_THRESHOLD = 0.25  # relaxed threshold for occluded/blurred hands


def _load_tflite(model_path: str):
    """Load a TFLite model via ai_edge_litert, falling back gracefully."""
    try:
        from ai_edge_litert.interpreter import Interpreter
        interp = Interpreter(model_path=model_path)
        interp.allocate_tensors()
        return interp
    except Exception as e:
        raise RuntimeError(f"Failed to load TFLite model {model_path}: {e}") from e


def _run_tflite(interp, inp_data: np.ndarray) -> List[np.ndarray]:
    """Run a TFLite interpreter and return all output tensors."""
    inp_details = interp.get_input_details()
    out_details = interp.get_output_details()
    interp.set_tensor(inp_details[0]['index'], inp_data)
    interp.invoke()
    return [interp.get_tensor(o['index']) for o in out_details]


def _preprocess_for_pose(image_bgr: np.ndarray) -> np.ndarray:
    """Resize + normalize for pose detection model (224x224 float32 [0,1])."""
    img = cv2.resize(image_bgr, (_POSE_DETECT_SIZE, _POSE_DETECT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img[np.newaxis]  # (1, 224, 224, 3)


def _preprocess_for_pose_landmark(image_bgr: np.ndarray) -> np.ndarray:
    """Resize + normalize for pose landmark model (256x256 float32 [0,1])."""
    img = cv2.resize(image_bgr, (_POSE_LANDMARK_SIZE, _POSE_LANDMARK_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img[np.newaxis]  # (1, 256, 256, 3)


def _preprocess_for_palm(image_bgr: np.ndarray) -> np.ndarray:
    """Resize + normalize for palm detection model (192x192 float32 [-1, 1])."""
    img = cv2.resize(image_bgr, (_PALM_DETECT_SIZE, _PALM_DETECT_SIZE))
    img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5) - 1.0
    return img[np.newaxis]  # (1, 192, 192, 3)


def _preprocess_for_hand(image_bgr: np.ndarray) -> np.ndarray:
    """Resize + normalize for hand landmark model (224x224 float32 [0,1])."""
    img = cv2.resize(image_bgr, (_HAND_MODEL_SIZE, _HAND_MODEL_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img[np.newaxis]  # (1, 224, 224, 3)


def _detect_palm_presence(palm_interp, image_bgr: np.ndarray) -> bool:
    """
    Run palm_detection_full.tflite on an image crop and return True if a palm
    is detected with score >= _PALM_SCORE_THRESHOLD.

    Model output:
      Identity   : (1, 2016, 18) — anchor box candidates
      Identity_1 : (1, 2016, 1) — raw logit scores (NOT sigmoid)
    """
    if palm_interp is None:
        return True  # no palm detector: always proceed (rely on landmark presence)
    inp = _preprocess_for_palm(image_bgr)
    outs = _run_tflite(palm_interp, inp)

    # Find the (1, 2016, 1) score tensor
    scores_raw = None
    for o in outs:
        if o.size == 2016:
            scores_raw = o.reshape(-1)
            break
    if scores_raw is None:
        return True  # can't parse — don't block

    # Sigmoid: presence = max(sigmoid(scores)) >= threshold
    max_score = float(np.max(1.0 / (1.0 + np.exp(-np.clip(scores_raw, -88, 88)))))
    return max_score >= _PALM_SCORE_THRESHOLD



def _extract_pose_landmarks(outputs: List[np.ndarray], H: int, W: int) -> Tuple[np.ndarray, bool]:
    """
    Parse pose landmark model outputs into (132,) feature vector.
    
    Pose landmark model outputs:
      [0]: (1, 195) = 39 landmarks × 5 (x, y, z, visibility, presence)
           in normalized [0,1] image coordinates
      [1]: (1, 1) = pose presence score
      [4]: (1, 117) = world landmarks 39 × 3
    """
    # The landmark output shape is (1, 195) → 39 × 5
    # We use only the first 33 landmarks (body) × 4 values (x,y,z,vis)
    raw = outputs[0].reshape(-1)  # (195,)
    presence_score = float(outputs[1].reshape(-1)[0])
    
    if presence_score < 0.3:
        return np.zeros(POSE_FEATURES, dtype=np.float32), False

    # Parse: 39 landmarks × 5 values each
    lms = raw.reshape(39, 5)[:33]  # first 33 = body landmarks
    # columns: x, y, z, visibility, presence
    pose = np.zeros((33, 4), dtype=np.float32)
    pose[:, 0] = lms[:, 0] / _POSE_LANDMARK_SIZE     # x (normalized 0-1)
    pose[:, 1] = lms[:, 1] / _POSE_LANDMARK_SIZE     # y (normalized 0-1)
    pose[:, 2] = lms[:, 2] / _POSE_LANDMARK_SIZE     # z (depth, normalize)
    pose[:, 3] = 1.0 / (1.0 + np.exp(-lms[:, 3]))    # visibility (sigmoid)

    return pose.flatten(), True


def _crop_hand_region(
    image_bgr: np.ndarray,
    wrist_xy: np.ndarray,
    elbow_xy: np.ndarray,
    scale: float = _HAND_CROP_SCALE,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Crop a square region around the wrist, sized proportional to
    elbow-to-wrist distance. Returns (cropped_image, bbox) where
    bbox = [x1, y1, x2, y2] in absolute pixels.
    """
    H, W = image_bgr.shape[:2]
    wx, wy = wrist_xy[0] * W, wrist_xy[1] * H
    ex, ey = elbow_xy[0] * W, elbow_xy[1] * H

    arm_len = np.sqrt((wx - ex) ** 2 + (wy - ey) ** 2) + 1e-6
    half = max(arm_len * scale / 2, 40)  # minimum 40px

    x1 = max(0, int(wx - half))
    y1 = max(0, int(wy - half))
    x2 = min(W, int(wx + half))
    y2 = min(H, int(wy + half))

    if (x2 - x1) < 20 or (y2 - y1) < 20:
        return None, np.array([x1, y1, x2, y2])

    crop = image_bgr[y1:y2, x1:x2]
    return crop, np.array([x1, y1, x2, y2])


def _extract_hand_landmarks(
    outputs: List[np.ndarray],
    bbox: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> Tuple[np.ndarray, bool]:
    """
    Parse hand landmark model outputs into (63,) feature vector.
    
    Hand landmark model outputs (hand_landmark_full.tflite):
      [0] Identity  → (1, 63) = 21 landmarks × 3 (x,y,z) in crop-relative coords
      [1] Identity_1 → (1, 1) = handedness score (≥0.5 = right hand)
      [2] Identity_2 → (1, 1) = hand presence score
      [3] Identity_3 → (1, 63) = world landmarks
    """
    # Find presence and landmark outputs by shape
    presence_score = 0.0
    landmarks_63 = None
    handedness_score = 0.5

    for o in outputs:
        flat = o.reshape(-1)
        if flat.shape[0] == 1 and landmarks_63 is not None:
            presence_score = float(flat[0])
        elif flat.shape[0] == 63 and landmarks_63 is None:
            landmarks_63 = flat
        elif flat.shape[0] == 1 and landmarks_63 is None:
            handedness_score = float(flat[0])

    # Re-parse properly using known index order
    # Output order: Identity (63), Identity_1 (1), Identity_2 (1), Identity_3 (63)
    sorted_out = sorted(outputs, key=lambda x: x.size)
    scores = [o.reshape(-1)[0] for o in sorted_out if o.size == 1]
    landmarks_tensors = [o.reshape(-1) for o in sorted_out if o.size == 63]

    if len(scores) >= 2:
        handedness_score = float(scores[0])
        presence_score = float(scores[1])
    elif len(scores) == 1:
        presence_score = float(scores[0])

    if len(landmarks_tensors) == 0 or presence_score < 0.3:
        return np.zeros(63, dtype=np.float32), False

    lms_crop = landmarks_tensors[0].reshape(21, 3)  # (21, 3) in crop coords [0,224]

    # Convert from crop-relative pixel coords → normalized frame coords
    x1, y1, x2, y2 = bbox
    crop_w = max(x2 - x1, 1)
    crop_h = max(y2 - y1, 1)

    lms_frame = np.zeros((21, 3), dtype=np.float32)
    lms_frame[:, 0] = (lms_crop[:, 0] / _HAND_MODEL_SIZE * crop_w + x1) / frame_w
    lms_frame[:, 1] = (lms_crop[:, 1] / _HAND_MODEL_SIZE * crop_h + y1) / frame_h
    lms_frame[:, 2] = lms_crop[:, 2] / _HAND_MODEL_SIZE  # z normalized

    return lms_frame.flatten(), True


# ---------------------------------------------------------------------------
# LandmarkExtractor — unified CPU-only TFLite backend
# ---------------------------------------------------------------------------

class LandmarkExtractor:
    """
    CPU-only MediaPipe landmark extractor using ai_edge_litert (TFLite).

    Runs pose detection → pose landmark → wrist-guided hand crop →
    hand landmark for each hand. Produces the same 258-feature vector
    as the legacy mp.solutions.holistic backend.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smooth_landmarks: bool = True,
        enable_segmentation: bool = False,
    ):
        self.min_detection_confidence = min_detection_confidence
        self._smooth = smooth_landmarks

        # Verify required model files exist
        for path in [_POSE_LANDMARK_MODEL, _HAND_LANDMARK_MODEL]:
            if not os.path.exists(path):
                raise RuntimeError(
                    f"[CRITICAL] TFLite model not found: {path}\n"
                    "Run the model download script first."
                )

        # Load TFLite interpreters
        self._pose_interp = _load_tflite(_POSE_LANDMARK_MODEL)
        self._hand_interp = _load_tflite(_HAND_LANDMARK_MODEL)

        # Palm detector: gates hand landmark calls, eliminating false positives
        # on blank/background frames. Optional — degrades gracefully if missing.
        self._palm_interp = None
        if os.path.exists(_PALM_DETECT_MODEL):
            try:
                self._palm_interp = _load_tflite(_PALM_DETECT_MODEL)
                print("[LandmarkExtractor] Palm detector loaded ✓")
            except Exception as e:
                print(f"[LandmarkExtractor] Palm detector load failed ({e}), skipping gate.")

        # Temporal smoothing: exponential moving average on landmark coords
        self._smooth_alpha = 0.5
        self._prev_pose: Optional[np.ndarray] = None
        self._prev_lh: Optional[np.ndarray] = None
        self._prev_rh: Optional[np.ndarray] = None

        self.use_legacy_solutions = False
        self.use_tasks_api = False
        self.use_tflite = True
        self.holistic = None  # legacy compat attribute

        print("[LandmarkExtractor] Using CPU TFLite backend (ai_edge_litert).")

    # ── Visibility tracking (used by process_frame in realtime_predict.py) ──

    @property
    def _last_results(self):
        return self.__last_results

    # ── Core extraction ──────────────────────────────────────────────────────

    def extract_frame_landmarks(self, image_bgr: np.ndarray, results=None) -> np.ndarray:
        """Extract 258-feature vector from a BGR frame."""
        fv, _ = self.process_frame_for_hud(image_bgr)
        return fv

    def process_frame_for_hud(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, "_ResultsAdapter"]:
        """
        Extract features and return (feature_vector, results_adapter).
        The results_adapter has .left_hand_landmarks and .right_hand_landmarks
        truthiness flags, compatible with realtime_predict.py's HUD and quality checks.
        """
        H, W = image_bgr.shape[:2]

        # ── 1. Pose landmarks ─────────────────────────────────────────────────
        pose_inp = _preprocess_for_pose_landmark(image_bgr)
        pose_outs = _run_tflite(self._pose_interp, pose_inp)
        pose_vec, pose_ok = _extract_pose_landmarks(pose_outs, H, W)

        if self._smooth and self._prev_pose is not None and pose_ok:
            pose_vec = self._smooth_alpha * pose_vec + (1 - self._smooth_alpha) * self._prev_pose
        if pose_ok:
            self._prev_pose = pose_vec.copy()

        # ── 2. Hand landmarks ─────────────────────────────────────────────────
        lh_vec = np.zeros(LEFT_HAND_FEATURES, dtype=np.float32)
        rh_vec = np.zeros(RIGHT_HAND_FEATURES, dtype=np.float32)
        lh_visible = False
        rh_visible = False

        if pose_ok:
            # Parse pose to (33,4) for wrist/elbow coordinates
            pose_33 = pose_vec.reshape(33, 4)

            for side, wrist_idx, elbow_idx in [
                ("left",  _LEFT_WRIST_IDX,  _LEFT_ELBOW_IDX),
                ("right", _RIGHT_WRIST_IDX, _RIGHT_ELBOW_IDX),
            ]:
                wrist_xy   = pose_33[wrist_idx,  :2]
                elbow_xy   = pose_33[elbow_idx,  :2]
                
                # We no longer check pose visibility score here. 
                # If the pose model guessed wrong, the palm detector will reject the crop anyway.

                # Use larger scale so crops catch hands resting in lap
                crop, bbox = _crop_hand_region(
                    image_bgr, wrist_xy, elbow_xy, scale=_HAND_CROP_SCALE
                )
                if crop is None:
                    continue

                # Gate: ensure palm detector confirms a hand exists here
                if not _detect_palm_presence(self._palm_interp, crop):
                    continue

                hand_inp  = _preprocess_for_hand(crop)
                hand_outs = _run_tflite(self._hand_interp, hand_inp)
                hand_vec, hand_ok = _extract_hand_landmarks(hand_outs, bbox, W, H)

                if not hand_ok:
                    continue

                if side == "left":
                    if self._smooth and self._prev_lh is not None:
                        hand_vec = self._smooth_alpha * hand_vec + (1 - self._smooth_alpha) * self._prev_lh
                    self._prev_lh = hand_vec.copy()
                    lh_vec    = hand_vec
                    lh_visible = True
                else:
                    if self._smooth and self._prev_rh is not None:
                        hand_vec = self._smooth_alpha * hand_vec + (1 - self._smooth_alpha) * self._prev_rh
                    self._prev_rh = hand_vec.copy()
                    rh_vec    = hand_vec
                    rh_visible = True

        # ── Full-frame fallback scan ──────────────────────────────────────────
        # When pose-guided crops find zero hands (hands in lap, arms not extended,
        # or pose model uncertain), scan large quadrant crops of the frame.
        # This catches hands that the wrist-guidance missed.
        if not lh_visible and not rh_visible:
            # Try center, quadrants, and full-frame
            scans = [
                (W//4,  H//4, 3*W//4, 3*H//4), # center (most likely)
                (0,     H//2, W//2, H),        # bottom-left
                (W//2,  H//2, W,   H),        # bottom-right
                (0,     0,    W//2, H//2),     # top-left
                (W//2,  0,    W,    H//2),     # top-right
                (0,     0,    W,    H),        # full-frame as last resort
            ]
            found = 0  # stop after finding 2 hands
            for (sx1, sy1, sx2, sy2) in scans:
                if found >= 2:
                    break
                if (sx2 - sx1) < 80 or (sy2 - sy1) < 80:
                    continue
                crop = image_bgr[sy1:sy2, sx1:sx2]
                bbox = np.array([sx1, sy1, sx2, sy2])
                
                # Gate: ensure palm detector confirms a hand exists in this region
                if not _detect_palm_presence(self._palm_interp, crop):
                    continue
                    
                hand_inp  = _preprocess_for_hand(crop)
                hand_outs = _run_tflite(self._hand_interp, hand_inp)
                hand_vec, hand_ok = _extract_hand_landmarks(hand_outs, bbox, W, H)

                if not hand_ok:
                    continue

                # Assign to whichever hand slot is still empty
                # (handedness from a full-frame scan is unreliable, so fill left first)
                if not lh_visible:
                    if self._smooth and self._prev_lh is not None:
                        hand_vec = self._smooth_alpha * hand_vec + (1 - self._smooth_alpha) * self._prev_lh
                    self._prev_lh = hand_vec.copy()
                    lh_vec     = hand_vec
                    lh_visible  = True
                elif not rh_visible:
                    if self._smooth and self._prev_rh is not None:
                        hand_vec = self._smooth_alpha * hand_vec + (1 - self._smooth_alpha) * self._prev_rh
                    self._prev_rh = hand_vec.copy()
                    rh_vec     = hand_vec
                    rh_visible  = True
                found += 1

        feature_vector = np.concatenate([pose_vec, lh_vec, rh_vec]).astype(np.float32)
        assert feature_vector.shape[0] == TOTAL_FEATURES, (
            f"Expected {TOTAL_FEATURES}, got {feature_vector.shape[0]}"
        )

        results = _ResultsAdapter(
            pose_visible=pose_ok,
            left_hand_visible=lh_visible,
            right_hand_visible=rh_visible,
        )
        return feature_vector, results

    # ── Video file processing (dataset pipeline) ─────────────────────────────

    def process_video(self, video_path: str, target_seq_len: int = SEQ_LEN) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        raw_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames.append(self.extract_frame_landmarks(frame))
        cap.release()

        if not raw_frames:
            return np.zeros((target_seq_len, TOTAL_FEATURES), dtype=np.float32)
        return self.normalize_sequence_length(np.array(raw_frames, dtype=np.float32), target_seq_len)

    @staticmethod
    def normalize_sequence_length(sequence: np.ndarray, target_seq_len: int = SEQ_LEN) -> np.ndarray:
        n = sequence.shape[0]
        if n == target_seq_len:
            return sequence
        if n > target_seq_len:
            idx = np.linspace(0, n - 1, target_seq_len, dtype=int)
            return sequence[idx]
        pad = np.zeros((target_seq_len - n, TOTAL_FEATURES), dtype=np.float32)
        return np.vstack([sequence, pad])

    def close(self):
        # TFLite interpreters don't need explicit close
        pass


# ---------------------------------------------------------------------------
# Results adapter (same interface as legacy holistic results)
# ---------------------------------------------------------------------------

class _ResultsAdapter:
    """Mimics mp.solutions.holistic result attributes for realtime_predict.py compatibility."""
    def __init__(self, pose_visible: bool, left_hand_visible: bool, right_hand_visible: bool):
        self.pose_landmarks = pose_visible or None
        self.left_hand_landmarks = left_hand_visible or None
        self.right_hand_landmarks = right_hand_visible or None


# ---------------------------------------------------------------------------
# Dataset pipeline helper
# ---------------------------------------------------------------------------

def process_dataset_directory(
    input_dir: str, output_dir: str, csv_output_path: str, target_seq_len: int = SEQ_LEN
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)

    extractor = LandmarkExtractor()
    records = []
    video_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_files.append((os.path.join(root, file), os.path.basename(root), file))

    print(f"Found {len(video_files)} video files in {input_dir}")
    for idx, (video_path, label, filename) in enumerate(tqdm(video_files, desc="Extracting")):
        video_id = os.path.splitext(filename)[0]
        try:
            landmarks = extractor.process_video(video_path, target_seq_len=target_seq_len)
            npy_path = os.path.join(output_dir, f"{label}_{video_id}_{idx}.npy")
            np.save(npy_path, landmarks)
            records.append({"video_id": video_id, "label": label, "npy_path": npy_path,
                            "frames": landmarks.shape[0], "features": landmarks.shape[1]})
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

    extractor.close()
    df = pd.DataFrame(records)
    df.to_csv(csv_output_path, index=False)
    print(f"Saved {len(records)} records to {csv_output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/raw/sample")
    parser.add_argument("--output_dir", default="data/landmarks")
    parser.add_argument("--csv_output", default="data/splits/labels.csv")
    args = parser.parse_args()
    if os.path.exists(args.input_dir):
        process_dataset_directory(args.input_dir, args.output_dir, args.csv_output)
    else:
        print(f"Input directory {args.input_dir} does not exist yet.")
