"""
ISL-Speak Configuration File
Centralized location for feature dimensions, sequence parameters, and training settings.
"""

# Feature Extraction Parameters
SEQ_LEN = 45  # Number of frames per video sequence

POSE_LANDMARKS = 33
POSE_VALS_PER_POINT = 4  # x, y, z, visibility
POSE_FEATURES = POSE_LANDMARKS * POSE_VALS_PER_POINT  # 132

HAND_LANDMARKS = 21
HAND_VALS_PER_POINT = 3  # x, y, z
LEFT_HAND_FEATURES = HAND_LANDMARKS * HAND_VALS_PER_POINT  # 63
RIGHT_HAND_FEATURES = HAND_LANDMARKS * HAND_VALS_PER_POINT  # 63

TOTAL_FEATURES = POSE_FEATURES + LEFT_HAND_FEATURES + RIGHT_HAND_FEATURES  # 258

# Sample default gesture classes for local testing / fallback
DEFAULT_CLASSES = [
    "hello",
    "thank_you",
    "please",
    "yes",
    "no",
    "help",
    "goodbye",
    "welcome",
    "water",
    "food"
]

# Model Parameters
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3

TRANSFORMER_D_MODEL = 128
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 3
TRANSFORMER_DIM_FEEDFORWARD = 512
TRANSFORMER_DROPOUT = 0.1

# Training Parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 60
PATIENCE = 10
TRAIN_VAL_TEST_SPLIT = (0.70, 0.15, 0.15)
