"""
Test Verification Script for FastAPI ISL-Speak Backend Endpoint.
Tests POST /predict using real .npy landmark files from data/landmarks/ and verifies output matches
the direct PyTorch RealtimeGesturePredictor result.
"""

import json
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from backend.main import app
from src.inference.realtime_predict import RealtimeGesturePredictor, clean_display_label


def test_fastapi_predict_endpoint():
    print("=== Testing FastAPI Backend Endpoint (/predict) ===")
    
    with TestClient(app) as client:
        health_resp = client.get("/health")
        assert health_resp.status_code == 200, f"Health check failed: {health_resp.text}"
        health_data = health_resp.json()
        print(f"✅ /health responded: status={health_data['status']}, device={health_data['device']}, classes={health_data['num_classes']}")
    
        labels_csv = os.path.join(PROJECT_ROOT, "data/splits/labels.csv")
        if not os.path.exists(labels_csv):
            print("⚠️ data/splits/labels.csv not found, skipping dataset test")
            return

        df = pd.read_csv(labels_csv)
        sample_rows = df.sample(min(5, len(df)), random_state=42)
        
        python_predictor = RealtimeGesturePredictor(
            checkpoint_path=os.path.join(PROJECT_ROOT, "checkpoints/best_lstm_model.pt"),
            mapping_json=os.path.join(PROJECT_ROOT, "data/splits/class_index_to_label.json"),
            model_type="lstm",
            enable_tts=False,
        )
        
        matches = 0
        total = 0
        
        print("\n--- Comparing FastAPI Response vs Direct PyTorch Inference ---")
        for _, row in sample_rows.iterrows():
            npy_path = os.path.join(PROJECT_ROOT, row["npy_path"]) if not os.path.isabs(row["npy_path"]) else row["npy_path"]
            if not os.path.exists(npy_path):
                continue

            landmarks = np.load(npy_path)
            if landmarks.shape != (45, 258):
                continue

            total += 1
            payload = {"sequence": landmarks.tolist(), "model_type": "lstm"}
            
            resp = client.post("/predict", json=payload)
            assert resp.status_code == 200, f"Predict failed: {resp.text}"
            api_result = resp.json()
            
            import torch
            seq_t = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(python_predictor.device)
            python_predictor.model.eval()
            with torch.no_grad():
                logits = python_predictor.model(seq_t)
                probs = torch.softmax(logits, dim=1)
                conf, idx = torch.max(probs, dim=1)
            
            py_class_idx = int(idx.item())
            py_conf = round(float(conf.item()), 4)
            py_label = clean_display_label(python_predictor.idx_to_label.get(py_class_idx, "Unknown"))

            is_match = (
                api_result["class_index"] == py_class_idx and
                api_result["label"] == py_label and
                abs(api_result["confidence"] - py_conf) < 1e-3
            )

            status_str = "✅ MATCH" if is_match else "❌ MISMATCH"
            if is_match:
                matches += 1

            print(f"[{status_str}] File: {os.path.basename(npy_path)}")
            print(f"   FastAPI : label='{api_result['label']}', conf={api_result['confidence']}, class_idx={api_result['class_index']}")
            print(f"   PyTorch : label='{py_label}', conf={py_conf}, class_idx={py_class_idx}")

        print(f"\nResult: {matches}/{total} exact matches between FastAPI & PyTorch!")
        assert matches == total, "FastAPI response must match direct PyTorch output!"
        print("✅ Phase 1 Verification Passed Successfully!")


if __name__ == "__main__":
    test_fastapi_predict_endpoint()
