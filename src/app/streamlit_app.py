"""
Camera-first ISL-Speak web prototype.

This screen intentionally behaves like the target mobile experience: open the
camera, keep the preview dominant, and show the live translation as a caption
below it. Research/demo features are kept out of the primary flow.
"""

import json
import os
import sys
import threading
import time
from typing import Dict, Tuple

import cv2
import streamlit as st

try:
    import av
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.realtime_predict import RealtimeGesturePredictor, clean_display_label


st.set_page_config(
    page_title="ISL Speak",
    page_icon=":camera:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --surface: #ffffff;
        --ink: #172026;
        --muted: #5d6973;
        --line: #d8dee4;
        --accent: #1565c0;
        --accent-soft: #e8f1fb;
        --success: #16794c;
        --warn: #9a5b00;
    }
    .stApp {
        background: #f6f8fa;
        color: var(--ink);
    }
    .block-container {
        max-width: 920px;
        padding-top: 24px;
        padding-bottom: 32px;
    }
    h1 {
        font-size: 28px !important;
        line-height: 1.2 !important;
        letter-spacing: 0 !important;
        margin-bottom: 4px !important;
    }
    .subtle {
        color: var(--muted);
        font-size: 15px;
        margin-bottom: 18px;
    }
    .status-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        border: 1px solid var(--line);
        background: var(--surface);
        border-radius: 8px;
        padding: 10px 12px;
        margin: 12px 0;
    }
    .status-pill {
        background: var(--accent-soft);
        color: var(--accent);
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 13px;
        font-weight: 700;
        white-space: nowrap;
    }
    .caption-panel {
        border: 1px solid var(--line);
        background: var(--surface);
        border-radius: 8px;
        padding: 16px;
        margin-top: 12px;
    }
    .caption-label {
        color: var(--muted);
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .04em;
        margin-bottom: 8px;
    }
    .caption-text {
        color: var(--ink);
        font-size: 28px;
        line-height: 1.28;
        font-weight: 700;
        min-height: 42px;
        overflow-wrap: anywhere;
    }
    .hint {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.45;
        margin-top: 8px;
    }
    .health {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.5;
    }
    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 8px 12px;
    }
    .stButton > button {
        border-radius: 8px;
        min-height: 42px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def checkpoint_for(model_choice: str) -> Tuple[str, str]:
    if model_choice == "Transformer":
        return "checkpoints/best_transformer_model.pt", "transformer"
    return "checkpoints/best_lstm_model.pt", "lstm"


@st.cache_resource(show_spinner=False)
def load_predictor(model_choice: str, confidence_threshold: float, enable_tts: bool):
    checkpoint_path, model_type = checkpoint_for(model_choice)
    return RealtimeGesturePredictor(
        checkpoint_path=checkpoint_path,
        mapping_json="data/splits/class_index_to_label.json",
        model_type=model_type,
        confidence_threshold=confidence_threshold,
        enable_tts=enable_tts,
    )


def load_vocabulary() -> list:
    mapping_file = "data/splits/class_index_to_label.json"
    if not os.path.exists(mapping_file):
        return []
    with open(mapping_file, "r") as f:
        return sorted(clean_display_label(v) for v in json.load(f).values())


def render_caption(status: str, word: str, confidence: float, sentence: str, quality: Dict) -> None:
    caption = sentence.strip() or word
    if status in {"warming_up", "hand_not_visible", "low_confidence", "scanning"} and not sentence.strip():
        caption = word
    caption = caption or "Point the camera at a clear ISL gesture"

    status_text = status.replace("_", " ").title()
    st.markdown(
        f"""
        <div class="status-row">
            <div>
                <strong>{word}</strong>
                <div class="hint">Confidence {confidence * 100:.0f}% | Hands visible: {"yes" if quality.get("hand_visible") else "no"}</div>
            </div>
            <div class="status-pill">{status_text}</div>
        </div>
        <div class="caption-panel">
            <div class="caption-label">Live translation</div>
            <div class="caption-text">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.title("ISL Speak")
st.markdown(
    '<div class="subtle">Live Indian Sign Language translation with a camera-first interface.</div>',
    unsafe_allow_html=True,
)

with st.expander("Detection settings", expanded=False):
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        model_choice = st.selectbox("Model", ["BiLSTM", "Transformer"], index=0)
    with col_b:
        detection_mode = st.selectbox("Mode", ["Balanced", "Faster", "Stricter"], index=0)
    with col_c:
        enable_tts = st.toggle("Speak captions", value=False)

threshold_by_mode = {
    "Faster": 0.45,
    "Balanced": 0.55,
    "Stricter": 0.70,
}
confidence_threshold = threshold_by_mode[detection_mode]

try:
    predictor = load_predictor(model_choice, confidence_threshold, enable_tts)
    predictor.confidence_threshold = confidence_threshold
    predictor.enable_tts = enable_tts
    if enable_tts and predictor.tts is None:
        from src.inference.tts_engine import TTSEngine

        predictor.tts = TTSEngine()
except Exception as exc:
    st.error(f"Detection cannot start: {exc}")
    st.stop()

if "caption_snapshot" not in st.session_state:
    st.session_state.caption_snapshot = {
        "status": "warming_up",
        "word": "Point the camera at a clear ISL gesture",
        "confidence": 0.0,
        "sentence": "",
        "quality": predictor.last_quality,
    }

controls = st.columns([1, 1, 1])
with controls[0]:
    if st.button("Clear caption", use_container_width=True):
        predictor.reset_buffer()
        st.session_state.caption_snapshot = {
            "status": "warming_up",
            "word": "Point the camera at a clear ISL gesture",
            "confidence": 0.0,
            "sentence": "",
            "quality": predictor.last_quality,
        }
with controls[1]:
    show_debug = st.toggle("Debug", value=False)
with controls[2]:
    st.write("")

caption_box = st.empty()


if WEBRTC_AVAILABLE:

    class GestureVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.lock = threading.Lock()
            self.status = "warming_up"
            self.word = "Point the camera at a clear ISL gesture"
            self.confidence = 0.0
            self.sentence = ""
            self.quality = predictor.last_quality

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            annotated, word, confidence, sentence = predictor.process_frame(image)
            with self.lock:
                self.status = predictor.last_status
                self.word = word
                self.confidence = confidence
                self.sentence = sentence
                self.quality = predictor.last_quality
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    ctx = webrtc_streamer(
        key="isl-speak-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=GestureVideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "facingMode": "user",
            },
            "audio": False,
        },
        async_processing=True,
    )

    render_caption(**st.session_state.caption_snapshot)

    while ctx.state.playing:
        processor = ctx.video_processor
        if processor:
            with processor.lock:
                st.session_state.caption_snapshot = {
                    "status": processor.status,
                    "word": processor.word,
                    "confidence": processor.confidence,
                    "sentence": processor.sentence,
                    "quality": processor.quality,
                }
            caption_box.empty()
            with caption_box.container():
                render_caption(**st.session_state.caption_snapshot)
        time.sleep(0.2)
else:
    st.warning("Live browser camera support is not installed. Using local camera fallback.")
    run_local = st.toggle("Start local camera", value=False)
    frame_box = st.empty()

    if run_local:
        cap = cv2.VideoCapture(0)
        while run_local and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                st.error("Unable to read from the camera.")
                break
            annotated, word, confidence, sentence = predictor.process_frame(frame)
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_box.image(frame_rgb, channels="RGB", use_container_width=True)
            caption_box.empty()
            with caption_box.container():
                render_caption(
                    predictor.last_status,
                    word,
                    confidence,
                    sentence,
                    predictor.last_quality,
                )
            time.sleep(0.03)
        cap.release()
    else:
        render_caption(**st.session_state.caption_snapshot)

st.markdown(
    '<div class="hint">For best results, keep one signer centered, keep hands inside the frame, and hold each gesture steady briefly.</div>',
    unsafe_allow_html=True,
)

if show_debug:
    health = predictor.health_report()
    vocab = load_vocabulary()
    st.divider()
    st.subheader("Debug")
    st.json(
        {
            "health": health,
            "last_quality": predictor.last_quality,
            "last_status": predictor.last_status,
            "supported_words": len(vocab),
            "first_words": vocab[:16],
        }
    )
