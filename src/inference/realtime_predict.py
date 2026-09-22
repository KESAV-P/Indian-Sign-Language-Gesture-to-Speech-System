"""
Real-Time Gesture Inference Pipeline.
Captures frames from webcam or video file, extracts MediaPipe Holistic landmarks,
maintains a rolling window deque of SEQ_LEN frames, evaluates PyTorch model checkpoint,
renders OpenCV HUD overlay, and streams accepted words to SentenceBuffer and TTSEngine.
"""

import os
import sys
import re
import json
import cv2
import torch
import numpy as np
from collections import deque
from typing import Optional, Dict, Tuple, List, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.extract_landmarks import LandmarkExtractor
from src.models.model_utils import build_model, load_checkpoint
from src.inference.sentence_buffer import SentenceBuffer
from src.inference.tts_engine import TTSEngine
from src.preprocessing.config import SEQ_LEN, TOTAL_FEATURES

try:
    import mediapipe.python as mp
except ImportError:
    import mediapipe as mp


def clean_display_label(label: str) -> str:
    """Removes numeric prefixes like '48. Hello' -> 'Hello'."""
    cleaned = re.sub(r"^\d+[\.\_\-\s]+", "", str(label)).strip()
    return cleaned.title() if cleaned else str(label).strip()


class RealtimeGesturePredictor:
    """
    Real-time sequence classifier pipeline.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_lstm_model.pt",
        mapping_json: str = "data/splits/class_index_to_label.json",
        model_type: str = "lstm",
        confidence_threshold: float = 0.60,
        window_stride: int = 3,
        enable_tts: bool = True,
    ):
        self.seq_len = SEQ_LEN
        self.total_features = TOTAL_FEATURES
        self.confidence_threshold = confidence_threshold
        self.window_stride = window_stride
        self.enable_tts = enable_tts
        self.frame_count = 0
        # Counts frames since last model evaluation (for non-overlapping window mode)
        self._frames_since_inference = 0
        self.last_predicted_word = "Waiting for gesture..."
        self.last_confidence = 0.0
        self.last_status = "warming_up"
        self.last_quality: Dict[str, Any] = {
            "pose_visible": False,
            "left_hand_visible": False,
            "right_hand_visible": False,
            "hand_visible": False,
            "sequence_fill": 0.0,
        }

        # Rolling landmark buffer
        self.landmark_window = deque(maxlen=SEQ_LEN)

        # Load class index to label mapping
        if os.path.exists(mapping_json):
            with open(mapping_json, "r") as f:
                raw_mapping = json.load(f)
            self.idx_to_label = {int(k): clean_display_label(v) for k, v in raw_mapping.items()}
        else:
            self.idx_to_label = {i: f"Gesture_{i}" for i in range(50)}

        self.num_classes = len(self.idx_to_label)

        # Load PyTorch Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(
            model_type,
            num_features=TOTAL_FEATURES,
            num_classes=self.num_classes,
            seq_len=SEQ_LEN,
        )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Model checkpoint not found: {checkpoint_path}. "
                "Live detection cannot run with untrained default weights."
            )

        self.model, _ = load_checkpoint(checkpoint_path, self.model, device=self.device)
        print(f"Loaded trained checkpoint from {checkpoint_path} ({self.num_classes} classes)")

        # Initialize MediaPipe, SentenceBuffer, and TTS
        self.extractor = LandmarkExtractor()
        # window_size=3: need only 3 model-stride observations to decide
        # min_frequency=2: 2 out of 3 must agree (tolerates 1 noisy frame)
        # repeat_cooldown=1: same word can re-fire after 1 different word
        # min_confidence matched to confidence_threshold so borderline
        # predictions (e.g. 0.45–0.50) are not silently discarded by the buffer.
        self.buffer = SentenceBuffer(
            window_size=4,           # 4 consecutive model observations
            min_confidence=0.60,     # accept at 60%+ confidence (prevents hallucinations)
            min_frequency=3,         # 3 out of 4 must agree
            repeat_cooldown=2,       # same word needs 2 different words between repeats
        )
        self.tts = TTSEngine() if enable_tts else None

    @staticmethod
    def landmark_quality(results, sequence_fill: float) -> Dict[str, Any]:
        """Return simple visibility signals for UI feedback and debugging."""
        pose_visible = bool(getattr(results, "pose_landmarks", None))
        left_hand_visible = bool(getattr(results, "left_hand_landmarks", None))
        right_hand_visible = bool(getattr(results, "right_hand_landmarks", None))
        hand_visible = left_hand_visible or right_hand_visible
        return {
            "pose_visible": pose_visible,
            "left_hand_visible": left_hand_visible,
            "right_hand_visible": right_hand_visible,
            "hand_visible": hand_visible,
            "sequence_fill": round(float(sequence_fill), 3),
        }

    def health_report(self) -> Dict[str, Any]:
        """Expose runtime readiness without requiring users to inspect logs."""
        ext = self.extractor
        mediapipe_ready = bool(
            getattr(ext, "holistic", None)          # legacy mp.solutions path
            or getattr(ext, "use_tasks_api", False)  # Tasks API path
            or getattr(ext, "use_tflite", False)     # CPU TFLite path
        )
        backend = (
            "tflite_cpu" if getattr(ext, "use_tflite", False) else
            "tasks_api"  if getattr(ext, "use_tasks_api", False) else
            "legacy_holistic" if getattr(ext, "holistic", None) else
            "unavailable"
        )
        return {
            "device": str(self.device),
            "classes": self.num_classes,
            "sequence_length": self.seq_len,
            "features": self.total_features,
            "confidence_threshold": self.confidence_threshold,
            "window_stride": self.window_stride,
            "mediapipe_ready": mediapipe_ready,
            "mediapipe_backend": backend,
            "tts_enabled": bool(self.enable_tts and self.tts),
        }

    def draw_landmarks_and_hud(
        self,
        frame_bgr: np.ndarray,
        results,
        word: str,
        confidence: float,
        sentence: str,
    ) -> np.ndarray:
        """
        Draws MediaPipe landmarks and modern HUD overlay onto frame.
        """
        annotated = frame_bgr.copy()
        h, w, _ = annotated.shape

        # 1. Draw MediaPipe landmarks skeleton
        if results is not None:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "drawing_utils"):
                mp_drawing = mp.solutions.drawing_utils
                mp_holistic = mp.solutions.holistic
                
                # Draw Pose
                if hasattr(results, "pose_landmarks") and results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 215, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 165, 255), thickness=2)
                    )
                # Draw Hands
                if hasattr(results, "left_hand_landmarks") and results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(121, 22, 254), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2)
                    )
                if hasattr(results, "right_hand_landmarks") and results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
                    )

        # 2. Top Header HUD Box (Glassmorphism dark overlay)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (15, 15), (w - 15, 105), (15, 17, 26), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
        cv2.rectangle(annotated, (15, 15), (w - 15, 105), (0, 210, 255), 1)

        # Status text & Recognized Word
        display_word = word.upper() if word and not word.startswith("Low") and not word.startswith("Wait") else word
        cv2.putText(annotated, f"GESTURE: {display_word}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Confidence Bar
        bar_x1, bar_y1, bar_x2, bar_y2 = 30, 65, w - 180, 85
        cv2.rectangle(annotated, (bar_x1, bar_y1), (bar_x2, bar_y2), (50, 50, 60), -1)
        
        bar_fill = int((bar_x2 - bar_x1) * min(max(confidence, 0.0), 1.0))
        color = (0, 255, 128) if confidence >= self.confidence_threshold else (0, 165, 255)
        if bar_fill > 0:
            cv2.rectangle(annotated, (bar_x1, bar_y1), (bar_x1 + bar_fill, bar_y2), color, -1)
            
        cv2.putText(annotated, f"{confidence * 100:.0f}%", (bar_x2 + 15, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

        # 3. Bottom Sentence Box
        overlay_bot = annotated.copy()
        cv2.rectangle(overlay_bot, (15, h - 75), (w - 15, h - 15), (15, 22, 36), -1)
        cv2.addWeighted(overlay_bot, 0.80, annotated, 0.20, 0, annotated)
        cv2.rectangle(annotated, (15, h - 75), (w - 15, h - 15), (0, 255, 160), 2)

        disp_sentence = f"SPEECH: {sentence}" if sentence else "SPEECH: Waiting for gestures..."
        cv2.putText(annotated, disp_sentence, (30, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 200), 2)

        return annotated

    def process_frame(
        self, frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, str, float, str]:
        """
        Processes a single video frame, extracts landmarks, runs PyTorch inference,
        and renders HUD overlay.
        """
        self.frame_count += 1

        # 1. MediaPipe feature extraction — works on both legacy and Tasks API paths
        feature_vector, results = self.extractor.process_frame_for_hud(frame_bgr)
        self.last_quality = self.landmark_quality(results, len(self.landmark_window) / float(self.seq_len))

        # ── HAND GUARD ──────────────────────────────────────────────────────────
        # Do NOT accumulate frames when no hand is visible. Feeding zero-hand
        # vectors into the model causes it to hallucinate high-confidence outputs
        # for whatever class best fits an all-zero hand signal (typically
        # "Mother", "Brother", etc. whose signs overlap with the zero baseline).
        if not self.last_quality["hand_visible"]:
            # Reset the window so stale zero-frames don't linger
            self.landmark_window.clear()
            self._frames_since_inference = 0
            current_pred_word = "Show your hand"
            confidence = 0.0
            self.last_status = "hand_not_visible"
            assembled_sentence = self.buffer.get_current_sentence()
            annotated_frame = self.draw_landmarks_and_hud(
                frame_bgr, results, current_pred_word, confidence, assembled_sentence
            )
            self.last_predicted_word = current_pred_word
            self.last_confidence = 0.0
            return annotated_frame, current_pred_word, confidence, assembled_sentence
        # ── END HAND GUARD ──────────────────────────────────────────────────────

        self.landmark_window.append(feature_vector)
        sequence_fill = len(self.landmark_window) / float(self.seq_len)
        self.last_quality = self.landmark_quality(results, sequence_fill)

        current_pred_word = self.last_predicted_word
        confidence = self.last_confidence
        if len(self.landmark_window) < self.seq_len:
            current_pred_word = f"Hold steady ({int(sequence_fill * 100)}%)"
            self.last_status = "warming_up"
        else:
            self.last_status = "scanning"

        # 2. Perform sequence classification — fire only when enough *new* frames
        # have accumulated since the last inference.
        self._frames_since_inference += 1
        run_inference = (
            len(self.landmark_window) == self.seq_len
            and self._frames_since_inference >= self.window_stride
        )
        if run_inference:
            self._frames_since_inference = 0
            seq_tensor = torch.tensor(
                np.array(self.landmark_window), dtype=torch.float32
            ).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(seq_tensor)
                probs = torch.softmax(logits, dim=1)
                conf, pred_class_idx = torch.max(probs, dim=1)

                confidence = conf.item()
                raw_pred_class = self.idx_to_label.get(pred_class_idx.item(), "Unknown")
                predicted_class = clean_display_label(raw_pred_class)

                if confidence >= self.confidence_threshold:
                    current_pred_word = predicted_class
                    self.last_status = "translated"
                    accepted_word = self.buffer.add_prediction(predicted_class, confidence)

                    if accepted_word and self.enable_tts and self.tts:
                        self.tts.speak(accepted_word, async_mode=True)
                else:
                    current_pred_word = f"Scanning ({confidence*100:.0f}%)"
                    self.last_status = "low_confidence"
                    # Low-confidence predictions do NOT vote — they are noise.

                self.last_predicted_word = current_pred_word
                self.last_confidence = confidence

            # Overlapping sliding window: drop only the oldest window_stride frames.
            for _ in range(min(self.window_stride, len(self.landmark_window))):
                self.landmark_window.popleft()

        assembled_sentence = self.buffer.get_current_sentence()

        # 3. Render HUD overlay and landmarks onto frame
        annotated_frame = self.draw_landmarks_and_hud(
            frame_bgr, results, current_pred_word, confidence, assembled_sentence
        )

        return annotated_frame, current_pred_word, confidence, assembled_sentence

    def reset_buffer(self):
        """Resets sliding window buffer state."""
        self.landmark_window.clear()
        self.buffer.clear()
        self.frame_count = 0
        self._frames_since_inference = 0
        self.last_predicted_word = "Waiting for gesture..."
        self.last_confidence = 0.0
        self.last_status = "warming_up"
        self.last_quality = {
            "pose_visible": False,
            "left_hand_visible": False,
            "right_hand_visible": False,
            "hand_visible": False,
            "sequence_fill": 0.0,
        }

    def close(self):
        self.extractor.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "transformer"])
    args = parser.parse_args()

    checkpoint = f"checkpoints/best_{args.model}_model.pt"
    predictor = RealtimeGesturePredictor(model_type=args.model, checkpoint_path=checkpoint)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
        
    print("Starting live inference... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
            
        annotated_frame, word, conf, sentence = predictor.process_frame(frame)
        cv2.imshow("ISL Gesture-to-Speech", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    predictor.close()
