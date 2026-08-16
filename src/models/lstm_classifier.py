"""
PyTorch 2-Layer Bidirectional LSTM Gesture Sequence Classifier.
Maps keypoint sequence (Batch, SEQ_LEN, num_features) to class logits (Batch, num_classes).
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.config import (
    SEQ_LEN,
    TOTAL_FEATURES,
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    LSTM_DROPOUT,
)



class SignLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM sequence classifier for Indian Sign Language landmarks.
    """

    def __init__(
        self,
        num_features: int = TOTAL_FEATURES,
        num_classes: int = 10,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        seq_len: int = SEQ_LEN,
    ):
        super(SignLSTMClassifier, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len

        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        
        # Linear classification head from concatenated forward+backward hidden states
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Shape (Batch, SEQ_LEN, num_features)
        Returns:
            torch.Tensor: Logits of shape (Batch, num_classes)
        """
        # x shape: (batch_size, seq_len, num_features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # h_n shape: (num_layers * num_directions, batch_size, hidden_size)
        # Extract last forward state and last backward state
        forward_final = h_n[-2, :, :]  # (batch_size, hidden_size)
        backward_final = h_n[-1, :, :] # (batch_size, hidden_size)
        
        # Concatenate forward and backward final states
        out = torch.cat((forward_final, backward_final), dim=1) # (batch_size, hidden_size * 2)
        out = self.dropout(out)
        logits = self.fc(out) # (batch_size, num_classes)
        
        return logits


if __name__ == "__main__":
    # Quick sanity check
    model = SignLSTMClassifier(num_features=258, num_classes=10)
    dummy_input = torch.randn(8, 45, 258)
    logits = model(dummy_input)
    print(f"SignLSTMClassifier Output Logits Shape: {logits.shape}")
    assert logits.shape == (8, 10), f"Expected (8, 10), got {logits.shape}"
