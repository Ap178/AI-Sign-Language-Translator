# training/train_lstm.py


import os
import numpy as np
import pickle


from tqdm import tqdm


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


from model import build_lstm_model



# ==========================
# CONFIG
# ==========================


DATASET_PATH = "/content/landmarks"

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 1530



MODEL_PATH = "/content/sign_lstm_model.h5"

ENCODER_PATH = "/content/label_encoder.pkl"



# ==========================
# LOAD DATA
# ==========================


X = []
y = []


classes = os.listdir(DATASET_PATH)


print("Classes:")
print(classes)



for label in classes:

    folder = os.path.join(
        DATASET_PATH,
        label
    )


    for file in tqdm(
        os.listdir(folder),
        desc=label
    ):

        if file.endswith(".npy"):


            path = os.path.join(
                folder,
                file
            )


            data = np.load(path)


            if data.shape == (
                SEQUENCE_LENGTH,
                FEATURE_SIZE
            ):

                X.append(data)
                y.append(label)



X = np.array(X)
y = np.array(y)



print("\nDataset")
print("X:",X.shape)
print("labels:",y.shape)



# ==========================
# ENCODE LABELS
# ==========================


encoder = LabelEncoder()


y_encoded = encoder.fit_transform(y)


with open(
    ENCODER_PATH,
    "wb"
) as f:

    pickle.dump(
        encoder,
        f
    )


y_encoded = to_categorical(
    y_encoded
)



NUM_CLASSES = y_encoded.shape[1]

print(
    "Classes:",
    NUM_CLASSES
)



# ==========================
# SPLIT
# ==========================


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y_encoded,

    test_size=0.2,

    random_state=42,

    stratify=y_encoded
)



# ==========================
# MODEL
# ==========================


model = build_lstm_model(

    SEQUENCE_LENGTH,

    FEATURE_SIZE,

    NUM_CLASSES

)


model.summary()



model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)



# ==========================
# CALLBACKS
# ==========================


early_stop = EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True,

    verbose=1

)



reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=5,

    verbose=1

)



checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1

)



# ==========================
# TRAIN
# ==========================



history = model.fit(

    X_train,

    y_train,

    validation_data=(

        X_test,

        y_test

    ),

    epochs=100,

    batch_size=16,

    callbacks=[

        early_stop,

        reduce_lr,

        checkpoint

    ]

)



print("Training completed")

print(
    "Saved:",
    MODEL_PATH
)