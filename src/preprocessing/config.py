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

# Model Parameters — BiLSTM (small: suits ≤200 real training samples)
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.5

# Model Parameters — Transformer (small)
TRANSFORMER_D_MODEL = 64
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_DIM_FEEDFORWARD = 128
TRANSFORMER_DROPOUT = 0.3

# Training Parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
NUM_EPOCHS = 200
PATIENCE = 25
TRAIN_VAL_TEST_SPLIT = (0.70, 0.15, 0.15)
