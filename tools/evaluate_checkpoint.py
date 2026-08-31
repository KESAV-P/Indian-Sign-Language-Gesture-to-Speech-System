"""
Evaluate an ISL-Speak checkpoint on saved landmark splits.

This is the fastest way to tell whether a detection problem is caused by the
trained classifier or by live camera/landmark capture.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, Tuple

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.realtime_predict import clean_display_label
from src.models.model_utils import build_model, load_checkpoint
from src.preprocessing.config import SEQ_LEN, TOTAL_FEATURES


def load_label_map(mapping_path: str) -> Dict[int, str]:
    with open(mapping_path, "r") as f:
        raw = json.load(f)
    return {int(k): clean_display_label(v) for k, v in raw.items()}


def infer_model_type(checkpoint_path: str) -> str:
    name = os.path.basename(checkpoint_path).lower()
    if "transformer" in name:
        return "transformer"
    return "lstm"


def evaluate(
    checkpoint_path: str,
    split_path: str,
    mapping_path: str,
    model_type: str,
    limit: int,
    batch_size: int,
) -> Tuple[float, Counter, list]:
    labels = load_label_map(mapping_path)
    data = np.load(split_path)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)

    if limit > 0:
        X = X[:limit]
        y = y[:limit]

    if X.ndim != 3 or X.shape[1:] != (SEQ_LEN, TOTAL_FEATURES):
        raise ValueError(f"Expected X shape (N, {SEQ_LEN}, {TOTAL_FEATURES}), got {X.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        model_type,
        num_features=TOTAL_FEATURES,
        num_classes=len(labels),
        seq_len=SEQ_LEN,
    )
    model, _ = load_checkpoint(checkpoint_path, model, device=device)

    correct = 0
    total = 0
    mistakes = Counter()
    examples = []
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            xb = torch.from_numpy(X[start:end]).to(device)
            yb = torch.from_numpy(y[start:end]).to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

            for true_idx, pred_idx, pred_conf in zip(yb.cpu().tolist(), pred.cpu().tolist(), conf.cpu().tolist()):
                total += 1
                is_correct = true_idx == pred_idx
                correct += int(is_correct)
                true_label = labels.get(true_idx, f"class_{true_idx}")
                pred_label = labels.get(pred_idx, f"class_{pred_idx}")
                per_class[true_label]["total"] += 1
                per_class[true_label]["correct"] += int(is_correct)
                if not is_correct:
                    mistakes[(true_label, pred_label)] += 1
                if len(examples) < 12:
                    examples.append(
                        {
                            "true": true_label,
                            "predicted": pred_label,
                            "confidence": round(float(pred_conf), 4),
                            "correct": is_correct,
                        }
                    )

    accuracy = correct / total if total else 0.0
    weak_classes = Counter()
    for label, stats in per_class.items():
        class_acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
        weak_classes[label] = round(class_acc, 4)

    return accuracy, mistakes, examples, weak_classes


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained gesture checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/best_lstm_model.pt")
    parser.add_argument("--split", default="data/splits/X_val.npz")
    parser.add_argument("--mapping", default="data/splits/class_index_to_label.json")
    parser.add_argument("--model", choices=["auto", "lstm", "transformer"], default="auto")
    parser.add_argument("--limit", type=int, default=0, help="Limit sample count. 0 means all.")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    model_type = infer_model_type(args.checkpoint) if args.model == "auto" else args.model
    accuracy, mistakes, examples, weak_classes = evaluate(
        args.checkpoint,
        args.split,
        args.mapping,
        model_type,
        args.limit,
        args.batch_size,
    )

    print(f"checkpoint: {args.checkpoint}")
    print(f"split: {args.split}")
    print(f"model_type: {model_type}")
    print(f"accuracy: {accuracy * 100:.2f}%")
    print("\nexamples:")
    for item in examples:
        marker = "OK" if item["correct"] else "MISS"
        print(f"  {marker} true={item['true']} predicted={item['predicted']} confidence={item['confidence']}")

    print("\ntop_mistakes:")
    for (true_label, pred_label), count in mistakes.most_common(12):
        print(f"  {true_label} -> {pred_label}: {count}")

    print("\nweakest_classes:")
    for label, class_acc in weak_classes.most_common()[:-13:-1]:
        print(f"  {label}: {class_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
