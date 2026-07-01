# Purpose: Keep all constants in one place.
import os
# =====================
# Project Paths
# =====================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)


RAW_VIDEO_DIR = os.path.join(
    DATA_DIR,
    "raw_videos"
)


LANDMARK_DIR = os.path.join(
    DATA_DIR,
    "landmarks"
)


MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sign_lstm_model.h5"
)


LABEL_ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "label_encoder.pkl"
)



# =====================
# MediaPipe Settings
# =====================


SEQUENCE_LENGTH = 30


MAX_HANDS = 2


HAND_FEATURE_SIZE = 126


# Face Mesh

FACE_LANDMARKS = 468


FACE_FEATURE_SIZE = FACE_LANDMARKS * 3



# Total feature size

TOTAL_FEATURE_SIZE = (
    HAND_FEATURE_SIZE +
    FACE_FEATURE_SIZE
)



# =====================
# Model Settings
# =====================

LSTM_UNITS_1 = 64

LSTM_UNITS_2 = 32

DROPOUT = 0.4