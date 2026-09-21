"""
FastAPI Backend Service for ISL-Speak Model Inference.
Serves PyTorch sign gesture models (BiLSTM & Transformer) over HTTP.

Endpoints:
  GET  /health           — server readiness check
  POST /predict          — accepts a pre-computed (45×258) landmark sequence → prediction
  POST /predict_frame    — accepts a raw JPEG/PNG image (base64), runs MediaPipe landmark
                           extraction on the server, maintains a per-session rolling 45-frame
                           window, fires LSTM inference every window_stride new frames, and
                           returns a real-time DetectionSnapshot compatible with the mobile app.
  POST /session/reset    — clears the landmark window + sentence buffer
"""

import base64
import io
import json
import os
import sys
import threading
from collections import deque
from typing import List, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.realtime_predict import clean_display_label
from src.inference.sentence_buffer import SentenceBuffer
from src.models.model_utils import build_model, load_checkpoint
from src.preprocessing.config import SEQ_LEN, TOTAL_FEATURES
from src.preprocessing.extract_landmarks import LandmarkExtractor

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ISL-Speak Inference Backend",
    description="REST API for real-time Indian Sign Language gesture translation",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic schemas ───────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    sequence: List[List[float]] = Field(
        ...,
        description="Sequence of 45 frames, each containing 258 float landmark features",
    )
    model_type: Optional[str] = Field(
        default="lstm",
        description="Model architecture to use: 'lstm' or 'transformer'",
    )


class PredictResponse(BaseModel):
    label: str
    confidence: float
    class_index: int
    status: str
    raw_label: str


class FrameRequest(BaseModel):
    """
    Image frame from the mobile camera, encoded as base64 JPEG/PNG.
    The mobile app sends this every ~200ms.
    """
    image_b64: str = Field(..., description="Base64-encoded JPEG/PNG camera frame")
    session_id: Optional[str] = Field(default="default", description="Session identifier")


class FramePredictResponse(BaseModel):
    """
    Real-time snapshot returned after each frame — mirrors DetectionSnapshot in the mobile app.
    """
    label: str              # Detected gesture label (e.g. "Hello")
    confidence: float       # Model confidence 0.0–1.0
    status: str             # "ready" | "hand_not_visible" | "warming_up" | "recognizing" | "translated"
    hands_visible: bool     # Whether MediaPipe found hands in this frame
    sentence: str           # Accumulated sentence so far
    sequence_fill: float    # How full the 45-frame window is (0.0–1.0)
    candidate: str          # Current candidate word (may differ from accepted label)


# ── ModelManager (for /predict — sequence-based API) ─────────────────────────


class ModelManager:
    """Manages PyTorch model instance and class mapping for the /predict endpoint."""

    def __init__(self, checkpoint_path: str = "checkpoints/best_lstm_model.pt", model_type: str = "lstm"):
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mapping_json = os.path.join(PROJECT_ROOT, "data/splits/class_index_to_label.json")

        if os.path.exists(self.mapping_json):
            with open(self.mapping_json, "r") as f:
                raw_mapping = json.load(f)
            self.idx_to_label = {int(k): v for k, v in raw_mapping.items()}
        else:
            self.idx_to_label = {i: f"Gesture_{i}" for i in range(52)}

        self.num_classes = len(self.idx_to_label)

        full_checkpoint_path = os.path.join(PROJECT_ROOT, checkpoint_path)
        if not os.path.exists(full_checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {full_checkpoint_path}")

        self.model = build_model(
            model_type,
            num_features=TOTAL_FEATURES,
            num_classes=self.num_classes,
            seq_len=SEQ_LEN,
        )
        self.model, _ = load_checkpoint(full_checkpoint_path, self.model, device=self.device)
        self.model.eval()

    def predict(self, sequence: np.ndarray, confidence_threshold: float = 0.45) -> PredictResponse:
        """Runs PyTorch model inference on a (45, 258) sequence."""
        if sequence.shape != (SEQ_LEN, TOTAL_FEATURES):
            raise ValueError(f"Invalid sequence shape {sequence.shape}, expected ({SEQ_LEN}, {TOTAL_FEATURES})")

        seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(seq_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred_class_idx = torch.max(probs, dim=1)

        confidence_val = float(conf.item())
        class_idx = int(pred_class_idx.item())
        raw_label = self.idx_to_label.get(class_idx, "Unknown")
        label = clean_display_label(raw_label)

        pred_status = "translated" if confidence_val >= confidence_threshold else "low_confidence"

        return PredictResponse(
            label=label,
            confidence=round(confidence_val, 4),
            class_index=class_idx,
            status=pred_status,
            raw_label=raw_label,
        )


# ── FrameSessionManager (for /predict_frame — real-time camera API) ───────────


class FrameSession:
    """
    Per-session state for real-time frame-by-frame inference.
    Holds the rolling 45-frame landmark window, sentence buffer, and last known state.
    Thread-safe via a session-level lock.
    """

    CONFIDENCE_THRESHOLD = 0.45
    WINDOW_STRIDE = 5  # Run inference every N new frames

    def __init__(self, model, idx_to_label: dict, device):
        self.model = model
        self.idx_to_label = idx_to_label
        self.device = device
        self.lock = threading.Lock()

        # Rolling 45-frame landmark window
        self.landmark_window: deque = deque(maxlen=SEQ_LEN)
        self._frames_since_inference: int = 0

        # Anti-flicker sentence buffer
        self.buffer = SentenceBuffer(
            window_size=5,
            min_confidence=0.60,
            min_frequency=4,
            repeat_cooldown=2,
        )

        # Persisted state between frames
        self.last_label: str = "Ready"
        self.last_confidence: float = 0.0
        self.last_status: str = "ready"

    def reset(self):
        with self.lock:
            self.landmark_window.clear()
            self._frames_since_inference = 0
            self.buffer.clear()
            self.last_label = "Ready"
            self.last_confidence = 0.0
            self.last_status = "ready"

    def process_feature_vector(
        self,
        feature_vector: np.ndarray,
        hands_visible: bool,
    ) -> FramePredictResponse:
        """
        Feed one 258-feature frame into the session's rolling window.
        Runs inference when enough new frames have accumulated.
        Returns a FramePredictResponse snapshot.
        """
        with self.lock:
            # ── Hand guard ────────────────────────────────────────────────────
            if not hands_visible:
                # Clear stale frames — zero-hand vectors cause hallucinations
                self.landmark_window.clear()
                self._frames_since_inference = 0
                self.last_status = "hand_not_visible"
                return FramePredictResponse(
                    label=self.last_label,
                    confidence=0.0,
                    status="hand_not_visible",
                    hands_visible=False,
                    sentence=self.buffer.get_current_sentence(),
                    sequence_fill=0.0,
                    candidate="Show your hand",
                )

            # ── Accumulate frame ──────────────────────────────────────────────
            self.landmark_window.append(feature_vector)
            sequence_fill = len(self.landmark_window) / float(SEQ_LEN)
            self._frames_since_inference += 1

            if len(self.landmark_window) < SEQ_LEN:
                self.last_status = "warming_up"
                return FramePredictResponse(
                    label=self.last_label,
                    confidence=self.last_confidence,
                    status="warming_up",
                    hands_visible=True,
                    sentence=self.buffer.get_current_sentence(),
                    sequence_fill=sequence_fill,
                    candidate=f"Collecting frames {int(sequence_fill * 100)}%",
                )

            # ── Inference (every window_stride new frames) ────────────────────
            run_inference = self._frames_since_inference >= self.WINDOW_STRIDE
            if run_inference:
                self._frames_since_inference = 0
                seq_arr = np.array(self.landmark_window, dtype=np.float32)
                seq_tensor = (
                    torch.tensor(seq_arr, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )

                with torch.no_grad():
                    logits = self.model(seq_tensor)
                    probs = torch.softmax(logits, dim=1)
                    conf, pred_idx = torch.max(probs, dim=1)

                confidence = float(conf.item())
                raw_label = self.idx_to_label.get(int(pred_idx.item()), "Unknown")
                predicted = clean_display_label(raw_label)

                if confidence >= self.CONFIDENCE_THRESHOLD:
                    self.last_label = predicted
                    self.last_confidence = confidence
                    self.last_status = "translated"
                    accepted = self.buffer.add_prediction(predicted, confidence)
                    candidate = predicted
                else:
                    self.last_status = "recognizing"
                    candidate = predicted

                # Slide window: drop oldest stride frames
                for _ in range(min(self.WINDOW_STRIDE, len(self.landmark_window))):
                    self.landmark_window.popleft()
            else:
                # Between inference cycles — show last known state
                candidate = self.last_label

            return FramePredictResponse(
                label=self.last_label,
                confidence=round(self.last_confidence, 4),
                status=self.last_status,
                hands_visible=True,
                sentence=self.buffer.get_current_sentence(),
                sequence_fill=sequence_fill,
                candidate=candidate,
            )


class FrameSessionManager:
    """
    Global registry of active FrameSession objects keyed by session_id.
    Lazy-creates sessions on first use.
    """

    def __init__(self, model, idx_to_label: dict, device):
        self.model = model
        self.idx_to_label = idx_to_label
        self.device = device
        self._sessions: dict = {}
        self._lock = threading.Lock()

    def get_session(self, session_id: str = "default") -> FrameSession:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = FrameSession(
                    self.model, self.idx_to_label, self.device
                )
            return self._sessions[session_id]

    def reset_session(self, session_id: str = "default"):
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].reset()


# ── Global singletons ──────────────────────────────────────────────────────────

model_manager: Optional[ModelManager] = None
extractor: Optional[LandmarkExtractor] = None
session_manager: Optional[FrameSessionManager] = None


@app.on_event("startup")
def startup_event():
    global model_manager, extractor, session_manager

    # Load PyTorch model (shared by both endpoints)
    model_manager = ModelManager(
        checkpoint_path="checkpoints/best_lstm_model.pt",
        model_type="lstm",
    )
    print(
        f"[Backend] PyTorch model loaded: {model_manager.num_classes} classes "
        f"on {model_manager.device}"
    )

    # Load MediaPipe TFLite extractor (used by /predict_frame)
    try:
        extractor = LandmarkExtractor()
        print("[Backend] MediaPipe TFLite landmark extractor loaded ✓")
    except Exception as e:
        print(f"[Backend] WARNING: MediaPipe extractor failed to load: {e}")
        print("[Backend] /predict_frame will be unavailable; /predict still works.")
        extractor = None

    # Create session manager (shares model weights with model_manager)
    session_manager = FrameSessionManager(
        model=model_manager.model,
        idx_to_label=model_manager.idx_to_label,
        device=model_manager.device,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.get("/health")
def health_check():
    if model_manager is None:
        raise HTTPException(status_code=500, detail="Model server not initialized")
    return {
        "status": "online",
        "device": str(model_manager.device),
        "num_classes": model_manager.num_classes,
        "checkpoint": model_manager.checkpoint_path,
        "sequence_len": SEQ_LEN,
        "total_features": TOTAL_FEATURES,
        "mediapipe_ready": extractor is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict_gesture(request: PredictRequest):
    """Sequence-based prediction: client sends a pre-computed (45×258) landmark array."""
    if model_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is initializing or unavailable",
        )

    seq_arr = np.array(request.sequence, dtype=np.float32)

    if seq_arr.shape != (SEQ_LEN, TOTAL_FEATURES):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Expected sequence of shape ({SEQ_LEN}, {TOTAL_FEATURES}), got {seq_arr.shape}",
        )

    try:
        response = model_manager.predict(seq_arr)
        return response
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}",
        )


@app.post("/predict_frame", response_model=FramePredictResponse)
def predict_frame(request: FrameRequest):
    """
    Real-time frame endpoint for the mobile app.

    The mobile client sends a base64-encoded JPEG camera frame every ~200ms.
    The server:
      1. Decodes the JPEG → BGR numpy array
      2. Runs MediaPipe TFLite landmark extraction (258 features)
      3. Feeds the feature vector into the session's rolling 45-frame window
      4. Fires PyTorch LSTM inference every 5 new frames
      5. Returns a FramePredictResponse snapshot

    The session state (landmark window + sentence buffer) is kept server-side,
    keyed by `session_id` (defaults to "default" for single-device use).
    """
    if extractor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MediaPipe extractor not available. Check TFLite model files.",
        )
    if session_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Session manager not initialized",
        )

    # ── Decode base64 image ───────────────────────────────────────────────────
    try:
        # Strip data URI prefix if present (e.g. "data:image/jpeg;base64,...")
        b64_data = request.image_b64
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]

        img_bytes = base64.b64decode(b64_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError("cv2.imdecode returned None")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode image: {e}",
        )

    # ── MediaPipe landmark extraction ─────────────────────────────────────────
    try:
        feature_vector, results = extractor.process_frame_for_hud(frame_bgr)
        hands_visible = bool(
            getattr(results, "left_hand_landmarks", None)
            or getattr(results, "right_hand_landmarks", None)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Landmark extraction error: {e}",
        )

    # ── Feed into session window + run inference ──────────────────────────────
    session = session_manager.get_session(request.session_id or "default")
    snapshot = session.process_feature_vector(feature_vector, hands_visible)
    return snapshot


@app.post("/session/reset")
def reset_session(session_id: str = "default"):
    """
    Clears the rolling landmark window and sentence buffer for a session.
    Call this when the user taps the Reset button in the app.
    """
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    session_manager.reset_session(session_id)
    return {"status": "reset", "session_id": session_id}
