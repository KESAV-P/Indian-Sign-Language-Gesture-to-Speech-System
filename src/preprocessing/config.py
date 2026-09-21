"""
ISL-Speak Configuration File
Centralized location for feature dimensions, sequence parameters, and training settings.
"""

# Feature Extraction Parameters
SEQ_LEN = 45  # Number of frames per video sequence (padded/trimmed to this length)

POSE_LANDMARKS = 33
POSE_VALS_PER_POINT = 4  # x, y, z, visibility
POSE_FEATURES = POSE_LANDMARKS * POSE_VALS_PER_POINT  # 132

HAND_LANDMARKS = 21
HAND_VALS_PER_POINT = 3  # x, y, z
LEFT_HAND_FEATURES = HAND_LANDMARKS * HAND_VALS_PER_POINT   # 63
RIGHT_HAND_FEATURES = HAND_LANDMARKS * HAND_VALS_PER_POINT  # 63

TOTAL_FEATURES = POSE_FEATURES + LEFT_HAND_FEATURES + RIGHT_HAND_FEATURES  # 258

# INCLUDE-50 gesture class vocabulary (real AI4Bharat INCLUDE dataset labels).
# These are the 50 high-frequency ISL words used in the INCLUDE-50 subset.
# The full INCLUDE dataset has 263 classes; this list is used only as a reference
# for class ordering when working with the INCLUDE-50 subset.
DEFAULT_CLASSES = [
    "After", "Agree", "All", "Always", "Bad",
    "Beautiful", "Before", "Better", "Boy", "Brother",
    "Buy", "Call", "Can", "College", "Come",
    "Come Back", "Cry", "Day", "Do Not Know", "Doctor",
    "Drink", "Eat", "Enjoy", "Everyone", "Father",
    "Feel", "Food", "Friend", "Girl", "Give",
    "Go", "Good", "Happy", "Home", "Hospital",
    "How", "Know", "Like", "Listen", "Love",
    "Meet", "Money", "Mother", "Name", "Need",
    "No", "Now", "People", "Please", "Right",
]

# Model Parameters — BiLSTM
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 3
LSTM_DROPOUT = 0.4

# Model Parameters — Transformer
TRANSFORMER_D_MODEL = 256
TRANSFORMER_NHEAD = 8
TRANSFORMER_NUM_LAYERS = 4
TRANSFORMER_DIM_FEEDFORWARD = 512
TRANSFORMER_DROPOUT = 0.2

# Training Parameters
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
PATIENCE = 15
TRAIN_VAL_TEST_SPLIT = (0.70, 0.15, 0.15)
