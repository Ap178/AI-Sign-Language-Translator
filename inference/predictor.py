# inference/predictor.py

import numpy as np
import pickle
import os

from keras.models import load_model


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "sign_lstm_model.h5")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

CONFIDENCE_THRESHOLD = 0.85


class SignPredictor:

    def __init__(self):
        print("Loading model...")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        if not os.path.exists(ENCODER_PATH):
            raise FileNotFoundError(f"Label encoder not found at {ENCODER_PATH}")

        self.model = load_model(MODEL_PATH, compile=False)

        with open(ENCODER_PATH, "rb") as f:
            self.encoder = pickle.load(f)

        print("Model loaded")

    def predict(self, sequence):
        """
        sequence: np.ndarray of shape (sequence_length, feature_dim)
        Returns: {"sign": str, "confidence": float}
        """
        if sequence is None:
            return {"sign": "Unknown", "confidence": 0.0}

        try:
            batch = np.expand_dims(sequence, axis=0)
            prediction = self.model.predict(batch, verbose=0)

            index = int(np.argmax(prediction[0]))
            confidence = float(prediction[0][index])

            if confidence < CONFIDENCE_THRESHOLD:
                return {"sign": "Unknown", "confidence": confidence}

            label = self.encoder.classes_[index]
            return {"sign": label, "confidence": confidence}

        except Exception as e:
            print(f"Prediction error: {e}")
            return {"sign": "Unknown", "confidence": 0.0}