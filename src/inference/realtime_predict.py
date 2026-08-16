"""
Real-Time Gesture Inference Pipeline.
Captures frames from webcam or video file, extracts MediaPipe Holistic landmarks,
maintains a rolling window deque of SEQ_LEN frames, evaluates PyTorch model checkpoint,
and streams accepted words to SentenceBuffer and TTSEngine.
"""

import os
import sys
import json
import cv2
import torch
import numpy as np
from collections import deque
from typing import Optional, Dict, Tuple, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.extract_landmarks import LandmarkExtractor
from src.models.model_utils import build_model, load_checkpoint
from src.inference.sentence_buffer import SentenceBuffer
from src.inference.tts_engine import TTSEngine
from src.preprocessing.config import SEQ_LEN, TOTAL_FEATURES


class RealtimeGesturePredictor:
    """
    Real-time sequence classifier pipeline.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_lstm_model.pt",
        mapping_json: str = "data/splits/class_index_to_label.json",
        model_type: str = "lstm",
        confidence_threshold: float = 0.6,
        window_stride: int = 5,
        enable_tts: bool = True,
    ):
        self.seq_len = SEQ_LEN
        self.total_features = TOTAL_FEATURES
        self.confidence_threshold = confidence_threshold
        self.window_stride = window_stride
        self.enable_tts = enable_tts
        self.frame_count = 0

        # Rolling landmark buffer
        self.landmark_window = deque(maxlen=SEQ_LEN)

        # Load class index to label mapping
        if os.path.exists(mapping_json):
            with open(mapping_json, "r") as f:
                raw_mapping = json.load(f)
            self.idx_to_label = {int(k): v for k, v in raw_mapping.items()}
        else:
            self.idx_to_label = {i: f"gesture_{i}" for i in range(10)}

        self.num_classes = len(self.idx_to_label)

        # Load PyTorch Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(
            model_type,
            num_features=TOTAL_FEATURES,
            num_classes=self.num_classes,
            seq_len=SEQ_LEN,
        )

        if os.path.exists(checkpoint_path):
            self.model, _ = load_checkpoint(checkpoint_path, self.model, device=self.device)
            print(f"Loaded trained checkpoint from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Running with uninitialized weights.")
            self.model.to(self.device)
            self.model.eval()

        # Initialize MediaPipe, SentenceBuffer, and TTS
        self.extractor = LandmarkExtractor()
        self.buffer = SentenceBuffer(window_size=5, min_confidence=confidence_threshold)
        self.tts = TTSEngine() if enable_tts else None

    def process_frame(
        self, frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, Optional[str], float, str]:
        """
        Processes a single video frame.
        
        Returns:
            Tuple: (annotated_frame_bgr, current_predicted_word, confidence_score, assembled_sentence)
        """
        self.frame_count += 1
        
        # Run MediaPipe feature extraction
        feature_vector = self.extractor.extract_frame_landmarks(frame_bgr)
        self.landmark_window.append(feature_vector)

        current_pred_word = "Waiting for gesture..."
        confidence = 0.0

        # Perform inference when buffer has reached full sequence length
        if len(self.landmark_window) == self.seq_len and (self.frame_count % self.window_stride == 0):
            seq_tensor = torch.tensor(
                np.array(self.landmark_window), dtype=torch.float32
            ).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(seq_tensor)
                probs = torch.softmax(logits, dim=1)
                conf, pred_class_idx = torch.max(probs, dim=1)

                confidence = conf.item()
                predicted_class = self.idx_to_label.get(pred_class_idx.item(), "unknown")

                if confidence >= self.confidence_threshold:
                    current_pred_word = predicted_class
                    accepted_word = self.buffer.add_prediction(predicted_class, confidence)

                    if accepted_word and self.enable_tts and self.tts:
                        self.tts.speak(accepted_word, async_mode=True)
                else:
                    current_pred_word = f"Low confidence ({confidence:.2f})"

        assembled_sentence = self.buffer.get_current_sentence()
        return frame_bgr, current_pred_word, confidence, assembled_sentence

    def reset_buffer(self):
        """Resets sliding window buffer state."""
        self.landmark_window.clear()
        self.buffer.clear()
        self.frame_count = 0

    def close(self):
        self.extractor.close()


if __name__ == "__main__":
    predictor = RealtimeGesturePredictor()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame, word, conf, sentence = predictor.process_frame(dummy_frame)
    print(f"Sample Frame Processed | Word: {word} | Conf: {conf:.2f} | Sentence: '{sentence}'")
    predictor.close()
