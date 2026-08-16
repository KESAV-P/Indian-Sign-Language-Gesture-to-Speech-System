"""
Training Script for ISL-Speak Sequence Classifiers.
Trains PyTorch BiLSTM or Transformer models with early stopping, learning rate scheduling, and checkpointing.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.model_utils import build_model, count_parameters, save_checkpoint
from src.preprocessing.config import (
    SEQ_LEN,
    TOTAL_FEATURES,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    PATIENCE,
)



def load_dataset(splits_dir: str):
    """Loads X_train, y_train, X_val, y_val, and class mapping."""
    train_data = np.load(os.path.join(splits_dir, "X_train.npz"))
    val_data = np.load(os.path.join(splits_dir, "X_val.npz"))

    X_train, y_train = train_data["X"], train_data["y"]
    X_val, y_val = val_data["X"], val_data["y"]

    mapping_json = os.path.join(splits_dir, "class_index_to_label.json")
    if os.path.exists(mapping_json):
        with open(mapping_json, "r") as f:
            class_mapping = json.load(f)
        num_classes = len(class_mapping)
    else:
        num_classes = len(np.unique(y_train))

    return X_train, y_train, X_val, y_val, num_classes


def train_model(
    model_type: str = "lstm",
    splits_dir: str = "data/splits",
    checkpoint_dir: str = "checkpoints",
    figures_dir: str = "reports/figures",
    batch_size: int = BATCH_SIZE,
    epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    patience: int = PATIENCE,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    X_train, y_train, X_val, y_val, num_classes = load_dataset(splits_dir)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using device: {device}")

    model = build_model(model_type, num_features=TOTAL_FEATURES, num_classes=num_classes, seq_len=SEQ_LEN)
    model.to(device)

    tot_p, trn_p = count_parameters(model)
    print(f"Instantiated {model_type.upper()} model | Total Params: {tot_p:,} | Trainable: {trn_p:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)


    best_val_acc = 0.0
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    checkpoint_filename = f"best_{model_type.lower()}_model.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_running_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == targets).sum().item()
                total_val += targets.size(0)

        epoch_val_loss = val_running_loss / total_val
        epoch_val_acc = correct_val / total_val

        scheduler.step(epoch_val_acc)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:.2f}% | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:.2f}%")

        # Save Best Checkpoint
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_type": model_type,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": best_val_acc,
                    "num_classes": num_classes,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs. Best Val Acc: {best_val_acc*100:.2f}%")
                break

    # Plot & Save Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history["train_loss"], label="Train Loss", color="royalblue")
    ax1.plot(history["val_loss"], label="Val Loss", color="orange")
    ax1.set_title(f"{model_type.upper()} Loss Curves")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot([acc * 100 for acc in history["train_acc"]], label="Train Acc (%)", color="royalblue")
    ax2.plot([acc * 100 for acc in history["val_acc"]], label="Val Acc (%)", color="orange")
    ax2.set_title(f"{model_type.upper()} Accuracy Curves")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(figures_dir, f"{model_type.lower()}_training_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved training curves to {plot_path}")

    return model, history, best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ISL gesture classification model.")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "transformer"], help="Model architecture")
    parser.add_argument("--splits_dir", type=str, default="data/splits", help="Directory containing X_train.npz, etc.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    train_model(model_type=args.model, splits_dir=args.splits_dir, epochs=args.epochs, batch_size=args.batch_size)
