# training/evaluate.py


import numpy as np
import pickle

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


from tensorflow.keras.models import load_model


MODEL_PATH = "/content/sign_lstm_model.h5"

ENCODER_PATH = "/content/label_encoder.pkl"



# Load model

model = load_model(
    MODEL_PATH
)


with open(
    ENCODER_PATH,
    "rb"
) as f:

    encoder = pickle.load(f)



# Load test data

X = np.load(
    "/content/X_test.npy"
)


y = np.load(
    "/content/y_test.npy"
)



prediction = model.predict(
    X
)



predicted_classes = np.argmax(
    prediction,
    axis=1
)


true_classes = np.argmax(
    y,
    axis=1
)



print(
    classification_report(

        true_classes,

        predicted_classes,

        target_names=encoder.classes_

    )
)



print(
    "Confusion Matrix"
)


print(
    confusion_matrix(

        true_classes,

        predicted_classes

    )
)