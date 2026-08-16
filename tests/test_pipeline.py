"""
Automated Unit Tests for ISL-Speak Pipeline.
"""

import os
import sys
import unittest
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.config import SEQ_LEN, TOTAL_FEATURES
from src.models.lstm_classifier import SignLSTMClassifier
from src.models.transformer_classifier import SignTransformerClassifier
from src.models.model_utils import build_model, count_parameters
from src.inference.sentence_buffer import SentenceBuffer
from src.inference.tts_engine import TTSEngine


class TestISLSpeakPipeline(unittest.TestCase):

    def test_config_dimensions(self):
        """Verify feature vector and sequence length constants."""
        self.assertEqual(SEQ_LEN, 45)
        self.assertEqual(TOTAL_FEATURES, 258)

    def test_lstm_forward_pass(self):
        """Test BiLSTM model instantiation and forward pass shapes."""
        model = SignLSTMClassifier(num_features=258, num_classes=10, seq_len=45)
        dummy_input = torch.randn(4, 45, 258)
        logits = model(dummy_input)
        self.assertEqual(logits.shape, (4, 10))

    def test_transformer_forward_pass(self):
        """Test Transformer Encoder model instantiation and forward pass shapes."""
        model = SignTransformerClassifier(num_features=258, num_classes=10, seq_len=45)
        dummy_input = torch.randn(4, 45, 258)
        logits = model(dummy_input)
        self.assertEqual(logits.shape, (4, 10))

    def test_model_factory(self):
        """Test build_model factory function."""
        lstm = build_model("lstm", num_features=258, num_classes=5)
        transformer = build_model("transformer", num_features=258, num_classes=5)
        
        tot_l, trn_l = count_parameters(lstm)
        tot_t, trn_t = count_parameters(transformer)
        
        self.assertGreater(tot_l, 0)
        self.assertGreater(tot_t, 0)

    def test_sentence_buffer_filtering(self):
        """Test SentenceBuffer majority voting and anti-flicker logic."""
        buf = SentenceBuffer(window_size=5, min_confidence=0.6)
        
        # Feed 5 consistent predictions of "hello"
        for _ in range(5):
            buf.add_prediction("hello", 0.90)

        self.assertEqual(buf.get_current_sentence(), "hello")

        # Feed consecutive duplicate "hello" - should not append
        for _ in range(5):
            buf.add_prediction("hello", 0.95)

        self.assertEqual(buf.get_current_sentence(), "hello")

        # Feed 5 consistent predictions of "thank_you"
        for _ in range(5):
            buf.add_prediction("thank_you", 0.88)

        self.assertEqual(buf.get_current_sentence(), "hello thank_you")

        flushed = buf.flush()
        self.assertEqual(flushed, "hello thank_you")
        self.assertEqual(buf.get_current_sentence(), "")

    def test_tts_engine_init(self):
        """Test non-blocking TTSEngine instantiation."""
        tts = TTSEngine()
        self.assertIsNotNone(tts)


if __name__ == "__main__":
    unittest.main()
