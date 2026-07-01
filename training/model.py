# Your dataset format:

# landmarks/
#  ├── hello/
#  │    ├── 001.npy
#  │    ├── 002.npy
#  │
#  ├── yes/
#  ├── no/

# Each .npy:

# (30,126)

# because:

# 30 frames
# 21 hand landmarks × xyz × 2 hands
# training/model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    BatchNormalization
)


def build_lstm_model(
        sequence_length,
        feature_size,
        num_classes
):

    model = Sequential()


    model.add(
        LSTM(
            128,
            return_sequences=True,
            input_shape=(
                sequence_length,
                feature_size
            )
        )
    )

    model.add(
        BatchNormalization()
    )

    model.add(
        Dropout(0.3)
    )


    model.add(
        LSTM(
            64
        )
    )


    model.add(
        Dense(
            128,
            activation="relu"
        )
    )


    model.add(
        Dropout(0.3)
    )


    model.add(
        Dense(
            num_classes,
            activation="softmax"
        )
    )


    return model