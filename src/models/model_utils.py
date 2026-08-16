"""
Model Utilities for ISL-Speak.
Provides model factory, parameter count, and checkpoint management utilities.
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.lstm_classifier import SignLSTMClassifier
from src.models.transformer_classifier import SignTransformerClassifier
from src.preprocessing.config import TOTAL_FEATURES, SEQ_LEN



def build_model(
    model_name: str = "lstm",
    num_features: int = TOTAL_FEATURES,
    num_classes: int = 10,
    seq_len: int = SEQ_LEN,
    **kwargs
) -> nn.Module:
    """
    Factory function to instantiate PyTorch gesture classification models.
    """
    model_name = model_name.lower().strip()
    if model_name in ["lstm", "bilstm", "signlstmclassifier"]:
        model = SignLSTMClassifier(
            num_features=num_features,
            num_classes=num_classes,
            seq_len=seq_len,
            **kwargs
        )
    elif model_name in ["transformer", "signtransformerclassifier"]:
        model = SignTransformerClassifier(
            num_features=num_features,
            num_classes=num_classes,
            seq_len=seq_len,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose 'lstm' or 'transformer'.")
    
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Returns total parameters and trainable parameters count.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_path: str
):
    """
    Saves PyTorch model checkpoint safely.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(state, checkpoint_path)
    print(f"Saved model checkpoint to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Loads PyTorch model checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model, checkpoint


if __name__ == "__main__":
    lstm_model = build_model("lstm", num_features=258, num_classes=10)
    trans_model = build_model("transformer", num_features=258, num_classes=10)

    l_tot, l_trn = count_parameters(lstm_model)
    t_tot, t_trn = count_parameters(trans_model)

    print(f"BiLSTM Total Params: {l_tot:,} | Trainable: {l_trn:,}")
    print(f"Transformer Total Params: {t_tot:,} | Trainable: {t_trn:,}")
