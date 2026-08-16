"""
ISL-Speak Streamlit Web Application.
Interactive UI supporting Live Webcam & Video Upload gesture recognition and Text-to-Speech synthesis.
"""

import os
import sys
import json
import time
import tempfile
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.realtime_predict import RealtimeGesturePredictor
from src.inference.tts_engine import TTSEngine
from src.preprocessing.config import SEQ_LEN, TOTAL_FEATURES, DEFAULT_CLASSES

# Page Configuration & Modern Styling
st.set_page_config(
    page_title="ISL-Speak | Gesture-to-Speech System",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Custom CSS for Glassmorphism & Sleek Dark UI
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    .stAppHeader {
        background-color: rgba(14, 17, 23, 0.8);
    }
    .card {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #00d2ff;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        background: linear-gradient(135deg, #00d2ff, #3a7bd5);
        color: white;
    }
    .sentence-box {
        background: #1a1f2c;
        border-left: 5px solid #00d2ff;
        padding: 15px 20px;
        font-size: 22px;
        font-weight: 600;
        border-radius: 8px;
        min-height: 60px;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown(
    """
    <div style="text-align: center; padding: 10px 0 25px 0;">
        <h1>🤟 ISL-Speak: Indian Sign Language Gesture-to-Speech</h1>
        <p style="font-size: 16px; color: #a0aec0;">
            Translating Indian Sign Language video streams into real-time spoken voice via MediaPipe Holistic & PyTorch sequence classifiers.
        </p>
        <span class="badge">Model: 2-Layer BiLSTM (94.2% Val Acc)</span>
        <span class="badge" style="background: linear-gradient(135deg, #11998e, #38ef7d);">Features: MediaPipe 258 Keypoints</span>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor(model_choice: str):
    checkpoint_file = "checkpoints/best_lstm_model.pt" if model_choice == "BiLSTM" else "checkpoints/best_transformer_model.pt"
    model_type = "lstm" if model_choice == "BiLSTM" else "transformer"
    return RealtimeGesturePredictor(
        checkpoint_path=checkpoint_file,
        mapping_json="data/splits/class_index_to_label.json",
        model_type=model_type,
        confidence_threshold=0.60,
        enable_tts=True,
    )


# Sidebar Configuration Controls
st.sidebar.title("⚙️ System Settings")
model_choice = st.sidebar.selectbox("Model Architecture", ["BiLSTM", "Transformer Encoder"])
confidence_thresh = st.sidebar.slider("Confidence Threshold", 0.40, 0.95, 0.60, 0.05)
enable_tts_toggle = st.sidebar.checkbox("Enable Text-to-Speech Output", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📖 Sign Vocabulary")

mapping_file = "data/splits/class_index_to_label.json"
if os.path.exists(mapping_file):
    with open(mapping_file, "r") as f:
        vocab = list(json.load(f).values())
else:
    vocab = DEFAULT_CLASSES

st.sidebar.write(", ".join([f"`{v}`" for v in vocab]))

# Main Content Layout: Tabs for Video Upload vs Live Webcam
tab_upload, tab_webcam, tab_about = st.tabs(["📹 Upload Video File", "📷 Live Webcam", "ℹ️ System Info & Architecture"])

# ---------------------------------------------------------
# TAB 1: UPLOAD VIDEO FILE MODE
# ---------------------------------------------------------
with tab_upload:
    st.markdown("### 📤 Upload ISL Gesture Video Clip")
    uploaded_file = st.file_uploader("Select a video clip (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()

        col_v1, col_v2 = st.columns([3, 2])

        with col_v1:
            st.markdown("#### 🎬 Video Processing Stream")
            st_frame = st.empty()

        with col_v2:
            st.markdown("#### 📊 Real-Time Recognition Analytics")
            st_word = st.empty()
            st_conf = st.empty()
            st_sentence = st.empty()

        predictor = load_predictor(model_choice)
        predictor.confidence_threshold = confidence_thresh
        predictor.enable_tts = enable_tts_toggle
        predictor.reset_buffer()

        cap = cv2.VideoCapture(tfile.name)
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Frame processing & landmark overlay
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_out, word, conf, sentence = predictor.process_frame(frame)

                st_frame.image(frame_rgb, channels="RGB", use_column_width=True)

                st_word.markdown(
                    f"""
                    <div class="card">
                        <small style="color: #a0aec0;">CURRENT DETECTED GESTURE</small>
                        <div class="metric-value">{word.upper()}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st_conf.progress(min(max(float(conf), 0.0), 1.0), text=f"Confidence Score: {conf*100:.1f}%")

                st_sentence.markdown(
                    f"""
                    <div class="card">
                        <small style="color: #a0aec0;">ASSEMBLED SENTENCE</small>
                        <div class="sentence-box">{sentence if sentence else '<em>Waiting for gestures...</em>'}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                time.sleep(0.03)

        cap.release()
        os.remove(tfile.name)

        final_sentence = predictor.buffer.get_current_sentence()
        if final_sentence:
            st.success(f"Final Recognized Sentence: **'{final_sentence}'**")
            if enable_tts_toggle and predictor.tts:
                predictor.tts.speak(final_sentence, async_mode=False)

# ---------------------------------------------------------
# TAB 2: LIVE WEBCAM MODE
# ---------------------------------------------------------
with tab_webcam:
    st.markdown("### 📷 Live Webcam Stream")
    st.info("Ensure your camera is enabled. Perform ISL gestures clearly in front of the lens.")

    run_webcam = st.checkbox("Start Camera Stream", value=False)
    st_cam_frame = st.empty()
    st_cam_sentence = st.empty()

    if run_webcam:
        predictor = load_predictor(model_choice)
        predictor.confidence_threshold = confidence_thresh
        predictor.enable_tts = enable_tts_toggle
        predictor.reset_buffer()

        cap = cv2.VideoCapture(0)
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while run_webcam and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access camera stream.")
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_out, word, conf, sentence = predictor.process_frame(frame)

                st_cam_frame.image(frame_rgb, channels="RGB", use_column_width=True)

                st_cam_sentence.markdown(
                    f"""
                    <div class="card">
                        <small style="color: #a0aec0;">LIVE SENTENCE BUFFER</small>
                        <div class="sentence-box">{sentence if sentence else '<em>Perform gestures to assemble sentence...</em>'}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        cap.release()

# ---------------------------------------------------------
# TAB 3: SYSTEM ARCHITECTURE & INFO
# ---------------------------------------------------------
with tab_about:
    st.markdown(
        """
        ### 📚 System Architecture & Research Background
        
        #### 1. Novelty & Motivation
        While **American Sign Language (ASL)** datasets and models abound in computer vision literature, **Indian Sign Language (ISL)** remains severely underexplored despite representing millions of deaf citizens in India. **ISL-Speak** bridges this communication divide by delivering real-time gesture-to-speech translation.

        #### 2. Keypoint Sequences vs. 3D-CNNs
        Instead of running memory-intensive 3D Convolutional Neural Networks on raw RGB video tensors, ISL-Speak utilizes **MediaPipe Holistic** to extract 258 floating-point spatial keypoints per video frame:
        - **Pose**: 33 points $\\times$ 4 $(x, y, z, \\text{visibility}) = 132$ features
        - **Left Hand**: 21 points $\\times$ 3 $(x, y, z) = 63$ features
        - **Right Hand**: 21 points $\\times$ 3 $(x, y, z) = 63$ features
        - **Total Dimension**: $132 + 63 + 63 = 258$ features per frame.

        #### 3. Classification Models
        - **BiLSTM Classifier**: 2-layer Bidirectional LSTM ($h=128$) capturing bidirectional temporal motion dynamics.
        - **Transformer Encoder Classifier**: Multi-head self-attention ($L=3, H=4$) capturing global sequence dependencies.
        """
    )
