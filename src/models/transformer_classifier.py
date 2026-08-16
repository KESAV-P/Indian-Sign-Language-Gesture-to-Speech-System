"""
PyTorch Transformer Encoder Gesture Sequence Classifier.
Maps keypoint sequence (Batch, SEQ_LEN, num_features) to class logits (Batch, num_classes).
"""

import os
import sys
import math
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.config import (
    SEQ_LEN,
    TOTAL_FEATURES,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_NHEAD,
    TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_DIM_FEEDFORWARD,
    TRANSFORMER_DROPOUT,
)



class SignTransformerClassifier(nn.Module):
    """
    Transformer Encoder sequence classifier for Indian Sign Language landmarks.
    """

    def __init__(
        self,
        num_features: int = TOTAL_FEATURES,
        num_classes: int = 10,
        d_model: int = TRANSFORMER_D_MODEL,
        nhead: int = TRANSFORMER_NHEAD,
        num_layers: int = TRANSFORMER_NUM_LAYERS,
        dim_feedforward: int = TRANSFORMER_DIM_FEEDFORWARD,
        dropout: float = TRANSFORMER_DROPOUT,
        seq_len: int = SEQ_LEN,
    ):
        super(SignTransformerClassifier, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.d_model = d_model
        self.seq_len = seq_len

        # Linear projection of input features to d_model space
        self.input_projection = nn.Linear(num_features, d_model)
        
        # Learned positional encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Shape (Batch, SEQ_LEN, num_features)
        Returns:
            torch.Tensor: Logits of shape (Batch, num_classes)
        """
        # x shape: (batch, seq_len, num_features)
        batch_size, seq_len, _ = x.shape
        
        # Project to d_model
        h = self.input_projection(x) # (batch, seq_len, d_model)
        
        # Add positional embedding
        h = h + self.pos_embedding[:, :seq_len, :]
        
        # Transformer encoder pass
        h_out = self.transformer_encoder(h) # (batch, seq_len, d_model)
        
        # Global mean pooling over sequence length dimension
        h_pooled = h_out.mean(dim=1) # (batch, d_model)
        h_pooled = self.dropout(h_pooled)
        
        logits = self.fc(h_pooled) # (batch, num_classes)
        return logits


if __name__ == "__main__":
    model = SignTransformerClassifier(num_features=258, num_classes=10)
    dummy_input = torch.randn(8, 45, 258)
    logits = model(dummy_input)
    print(f"SignTransformerClassifier Output Logits Shape: {logits.shape}")
    assert logits.shape == (8, 10), f"Expected (8, 10), got {logits.shape}"
