# ISL-Speak: Indian Sign Language Gesture-to-Speech System

**Author**: Kesav, B.E. CSE (AI & ML), Chennai Institute of Technology  
**Development Environment**: Google Antigravity IDE (Local) & Google Colab (GPU Model Training)

---

## 📌 Motivation & Novelty Angle
American Sign Language (ASL) has numerous mature open-source gesture recognition projects and public datasets. In contrast, **Indian Sign Language (ISL)** remains significantly underexplored despite India hosting one of the world's largest deaf and hard-of-hearing populations.

**ISL-Speak** addresses this accessibility gap by translating video input of ISL gestures directly into audible speech (Text-to-Speech). By utilizing **MediaPipe Holistic landmark extraction**, the raw video sequence is converted into a compact sequence of keypoint coordinates `(SEQ_LEN=45, num_features=258)`. This keypoint-based representation drastically reduces memory and compute footprints compared to heavy 3D-CNNs, enabling fast and scalable model training on Google Colab free-tier GPU while preserving crucial motion dynamics.

---

## 🏗️ System Architecture

```mermaid
flowchart LR
    A[Video Input / Live Webcam] --> B[MediaPipe Holistic Processing]
    B --> C[Extract 258 Keypoint Features / Frame]
    C --> D[Uniform Temporal Sampling SEQ_LEN=45]
    D --> E[PyTorch Classifier: BiLSTM / Transformer]
    E --> F[Predicted Gesture Label]
    F --> G[SentenceBuffer: Anti-Flicker Majority Voting]
    G --> H[TTSEngine: pyttsx3 / gTTS Audio Output]
```

---

## 📁 Repository Structure

```
isl-speak/
├── data/
│   ├── raw/                  # Downloaded INCLUDE gesture videos (gitignored)
│   ├── landmarks/            # Extracted per-video .npy feature vectors (gitignored)
│   └── splits/               # Pre-packaged X_train.npz, y_train.npz, labels.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Local data analysis on sample clips
│   ├── 02_landmark_extraction.ipynb # Colab notebook for mass MediaPipe extraction
│   ├── 03_model_training.ipynb     # Colab notebook for GPU model training
│   └── 04_evaluation.ipynb         # Colab notebook for test evaluation & confusion matrix
├── src/
│   ├── preprocessing/
│   │   ├── config.py         # Shared feature dimensions & hyperparameters
│   │   ├── extract_landmarks.py # MediaPipe Holistic feature extractor
│   │   └── build_dataset.py  # Local synthetic sample builder & npz packager
│   ├── models/
│   │   ├── lstm_classifier.py    # 2-Layer Bidirectional LSTM classifier
│   │   ├── transformer_classifier.py # Transformer Encoder sequence classifier
│   │   └── model_utils.py    # Model factory, parameter counter, & loading utilities
│   ├── inference/
│   │   ├── tts_engine.py     # Offline (pyttsx3) & online (gTTS) text-to-speech
│   │   ├── sentence_buffer.py # Majority voting & anti-flicker sentence buffer
│   │   └── realtime_predict.py # Real-time prediction pipeline from webcam stream
│   └── app/
│       └── streamlit_app.py  # Interactive Streamlit Web Application
├── checkpoints/              # Downloaded best PyTorch checkpoints (.pt)
├── reports/
│   ├── figures/              # Training loss curves & confusion matrix plots
│   └── final_report.md       # Full academic project report
├── requirements.txt          # Local environment dependencies
├── requirements_colab.txt    # Google Colab GPU environment dependencies
└── README.md
```

---

## 🚀 Setup & Execution Guide

### 1️⃣ Local Setup (Google Antigravity IDE)
1. Initialize virtual environment and install local dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Generate synthetic sample dataset for local testing:
   ```bash
   python3 src/preprocessing/build_dataset.py
   ```
3. Run local unit tests to verify feature shapes & model forward passes:
   ```bash
   python3 -m unittest discover tests
   ```

### 2️⃣ Model Training (Google Colab GPU)
> ⚠️ **Important**: Model training is optimized for Google Colab GPU (T4 runtime) to avoid local machine resource exhaustion.
1. Mount Google Drive and upload `notebooks/03_model_training.ipynb`.
2. Connect to T4 GPU runtime (`Runtime -> Change runtime type -> T4 GPU`).
3. Execute `03_model_training.ipynb` to train the BiLSTM / Transformer model.
4. Download the resulting `best_lstm_model.pt` checkpoint to your local `checkpoints/` directory.

### 3️⃣ Launching the Streamlit Demo App
Run the interactive application locally:
```bash
streamlit run src/app/streamlit_app.py
```
- **Webcam Mode**: Perform gestures in front of your camera for real-time classification and speech synthesis.
- **Upload Mode**: Upload pre-recorded ISL video clips (`.mp4`, `.avi`) for evaluation and demonstration.

---

## 📊 Results Summary

> ⚠️ **Data Quality Note**: An earlier version of this README reported 94.2% / 91.9% accuracy. Those numbers were computed on partially mislabeled and partially synthetic data and have been **retracted**. The numbers below are from the corrected, 100%-real, 193-sample dataset. See [reports/final_report.md](reports/final_report.md) Section 6.3 for the full data quality fix writeup.

**Dataset**: 193 real ISL signer videos, 11 classes — 135 train / 29 val / 29 test (all source = "real").

| Model Architecture | Input Shape | Val Accuracy | Test Accuracy | Params |
| :--- | :--- | :--- | :--- | :--- |
| **2-Layer BiLSTM** | `(Batch, 45, 258)` | 31.0% | 31.0% | 4.2M |
| **Transformer Encoder** | `(Batch, 45, 258)` | 48.3% | 34.5% | 2.2M |

**Root cause of low accuracy**: avg. ~12 real training samples per class after the 70/15/15 split. Classes with distinct hand shapes (Time, Hello) score well; visually similar pairs (Morning/Afternoon/Evening) remain confused. Collecting ≥50 samples per class is the primary next step.

### ✅ Data Integrity — `tools/validate_dataset.py`

Run before any retraining to catch labeling bugs automatically:

```bash
python tools/validate_dataset.py --min-samples 10
# All checks passed — dataset is clean and ready for training.
```

Checks: per-class counts · source audit (blocks synthetic) · cross-label cosine-similarity duplicates · category-folder-name guard.
