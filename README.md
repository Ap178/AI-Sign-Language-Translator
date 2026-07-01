---
title: AI Sign Language Translator
emoji: 🤟
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# 🤟 AI Sign Language Translator

### Real-Time Sign Language Recognition System using Computer Vision, LSTM Deep Learning & Text-to-Speech

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-red)
![Gradio](https://img.shields.io/badge/Gradio-HuggingFace-yellow)

---

## 🚀 Overview

AI Sign Language Translator is a real-time gesture recognition application that converts hand sign gestures into text and speech.

The system combines:

- MediaPipe for real-time hand landmark extraction
- A feature fusion pipeline for gesture representation
- An LSTM neural network for temporal gesture understanding
- Confidence-based prediction filtering and temporal voting for stability
- Text-to-Speech output
- A Gradio interface for web deployment on Hugging Face Spaces

The goal is to bridge communication gaps between deaf/mute communities and hearing individuals using AI.

---

## ✨ Features

### 🖐 Real-Time Gesture Detection

The application doesn't just classify every frame — it detects when a gesture actually starts and ends:

1. Detects hand landmarks every frame via MediaPipe
2. Watches for sustained motion to mark the **start** of a gesture
3. Buffers a rolling sequence of frames (with a pre-roll window so the very first frames of the gesture aren't lost)
4. Detects when the hand goes still to mark the **end** of a gesture
5. Runs LSTM inference on the completed sequence
6. Applies confidence filtering and temporal voting before committing to a prediction
7. Displays the result and speaks it aloud

```
Input:  Hand movement sequence
Output: "Hello"
Speech: 🔊 Hello
```

### 🧠 Deep Learning Model

The recognition model is based on an LSTM architecture.

**Why LSTM?** Sign language isn't a static image problem — a gesture is defined by hand position, direction, movement speed, and transition patterns over time. An LSTM is built to learn exactly these time-dependent relationships, rather than just classifying a single frame.

**Architecture:**

```
Input (30 frames × feature vector)
        |
        v
   LSTM Layer
        |
        v
   Dense Layer
        |
        v
Softmax Classification
        |
        v
   Sign Prediction
```

---

## 🏗 System Architecture

```
Webcam
  |
  v
OpenCV / Gradio Frame Capture
  |
  v
MediaPipe Hand Detection
  |
  v
Landmark Extraction
  |
  v
Feature Fusion
  |
  v
Gesture Start/End Detection + Buffering
  |
  v
LSTM Model Inference
  |
  v
Confidence Filtering + Temporal Voting
  |
  +-------------------+
  |                   |
  v                   v
Text Output      Text-to-Speech
```

---

## 📂 Project Structure

```
AI-Sign-Language-Translator/
│
├── app.py                      # Gradio web app (Hugging Face entry point)
│
├── models/
│   ├── sign_lstm_model.h5      # Trained LSTM model
│   └── label_encoder.pkl       # Label encoder for sign classes
│
├── dataset/
│   └── landmarks/              # Recorded gesture samples
│
├── features/
│   ├── hand_detection.py       # MediaPipe hand landmark detection
│   └── feature_fusion.py       # Per-frame feature vector construction
│
├── inference/
│   ├── camera.py                # Local OpenCV real-time inference loop
│   ├── predictor.py             # LSTM model loading + prediction
│   ├── gesture_buffer.py        # Rolling sequence buffer with pre-roll support
│   └── gesture_detector.py      # Motion-based gesture start/end detection
│
├── training/
│   ├── train_lstm.py            # Model training script
│   ├── model.py                 # Model architecture definition
│   └── record_new_sign.py       # Tool for recording new gesture samples
│
└── speech/
    └── tts.py                   # Text-to-Speech wrapper
```

---

## 🔥 AI Pipeline

### 1. Hand Detection

MediaPipe extracts 21 hand landmarks per frame, covering the wrist, fingers, joints, and palm.

### 2. Feature Engineering

Each frame is converted into a fixed-length feature vector capturing landmark positions (and, depending on configuration, derived features like relative distances or angles).

A full gesture sequence is represented as:

```
sequence_length frames × feature_dim per frame
```

### 3. Gesture Segmentation

Rather than predicting on a fixed window of frames, the system actively detects when a gesture begins (sustained motion) and ends (sustained stillness), with:

- A **pre-roll buffer** that captures the few frames just before motion is officially confirmed, so the start of the gesture isn't lost to detection lag
- A **minimum recording duration** before end-of-gesture can even be checked, so brief pauses mid-sign aren't mistaken for the gesture finishing
- A hard cap at the full sequence length as a safety fallback

### 4. Sequence Modeling

The LSTM learns movement direction, gesture timing, and motion patterns across the buffered sequence.

---

## 🛡 Anti-Hallucination System

To avoid unreliable predictions, the pipeline applies two layers of filtering:

### Confidence Threshold

Predictions below a configured confidence threshold (e.g. 85%) are rejected and returned as:

```
Unknown
```

### Temporal Voting

Rather than trusting a single prediction outright, recent predictions are tracked and the majority vote is used:

```
Prediction History: hello, hello, hello, yes, hello
Final Prediction:   hello
```

This reduces flicker and one-off misclassifications.

---

## 🎤 Text-to-Speech

Detected signs are converted into spoken audio, with a cooldown so the same sign isn't repeatedly spoken in quick succession.

```
Detected: Thank you
Output:   🔊 Thank you
```

---

## 🆕 Adding New Signs

The system supports collecting new gesture classes:

```
Record Gesture
      |
      v
Save Landmark Sequence
      |
      v
Train Model
      |
      v
New Sign Available
```

New samples are stored as:

```
dataset/landmarks/
  hello/
    0.npy
    1.npy
  yes/
    0.npy
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <repo-url>
cd AI-Sign-Language-Translator
```

Create an environment:

```bash
conda create -n signlang python=3.11
conda activate signlang
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Locally

**Web interface (Gradio):**

```bash
python app.py
```

Then open:

```
http://localhost:7860
```

**Native OpenCV window:**

```bash
python -m inference.camera
```

---

## 📦 Training

Prepare your landmark dataset under:

```
dataset/landmarks/
```

Then train:

```bash
python training/train_lstm.py
```

The trained model is saved to:

```
models/sign_lstm_model.h5
```

---

## ☁️ Hugging Face Deployment

This project is deployable on Hugging Face Spaces using the Gradio SDK.

Files required for deployment:

```
app.py
requirements.txt
models/
speech/
features/
inference/
```

---

## 📊 Currently Supported Signs

```
hello
yes
no
thank you
help
what
I
```

---

## 🔮 Future Improvements

- **Incremental learning** — add new signs without retraining the full model
- **Transformer-based model** — replace the LSTM with a Transformer encoder + attention mechanism for longer-range temporal context
- **Sentence-level translation** — chain multiple recognized signs into natural language sentences
- **Mobile deployment** — convert the model to TensorFlow Lite for an Android application

---

## 🧑‍💻 Tech Stack

| Component           | Technology           |
|----------------------|----------------------|
| Language             | Python                |
| Computer Vision       | OpenCV                |
| Landmark Detection    | MediaPipe             |
| Deep Learning         | TensorFlow / Keras    |
| Model                 | LSTM                  |
| UI                    | Gradio                |
| Deployment            | Hugging Face Spaces   |
| Speech                | gTTS              |

---

## 👨‍💻 Author

**Alby Ponnachan**
AI / Machine Learning Enthusiast

Interested in:
- Computer Vision
- Deep Learning
- AI Applications
- Cybersecurity

---

⭐ If you like this project, consider giving it a star!