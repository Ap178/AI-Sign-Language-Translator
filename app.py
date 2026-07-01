# app.py

import gradio as gr
import cv2
import time
from collections import deque, Counter

from inference.predictor import SignPredictor
from inference.gesture_buffer import GestureBuffer
from inference.gesture_detector import GestureStartDetector
from features.feature_fusion import FeatureExtractor
from speech.tts import TextToSpeech


# -----------------------------
# Load components
# -----------------------------
print("Loading AI model...")

predictor = SignPredictor()
extractor = FeatureExtractor()
tts = TextToSpeech()

# ---- tunables (same as camera.py) ----
SEQUENCE_LENGTH = 30
MIN_FRAMES = 15
PRE_ROLL_FRAMES = 10
SPEAK_COOLDOWN = 1.5
VOTE_HISTORY = 5

STATE_IDLE = "idle"
STATE_RECORDING = "recording"


class GestureSession:
    """
    Holds all per-session state. Gradio's streaming callback is just a
    function, so this groups everything that used to be loose globals
    into one object - and gives us a clean way to support multiple
    concurrent users later if needed.
    """

    def __init__(self):
        self.buffer = GestureBuffer(sequence_length=SEQUENCE_LENGTH, min_frames=MIN_FRAMES)
        self.detector = GestureStartDetector()
        self.pre_roll = deque(maxlen=PRE_ROLL_FRAMES)
        self.prediction_history = deque(maxlen=VOTE_HISTORY)

        self.state = STATE_IDLE
        self.last_spoken = ""
        self.last_spoken_time = 0.0

    def reset(self):
        self.buffer.clear()
        self.pre_roll.clear()
        self.prediction_history.clear()
        self.detector.reset()
        self.state = STATE_IDLE


session = GestureSession()


# -----------------------------
# Prediction function
# -----------------------------
def predict_sign(video_frame):
    if video_frame is None:
        return "No frame", None

    frame = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)
    features = extractor.extract(frame)
    hand_visible = features is not None

    # ==========================
    # IDLE: waiting for a gesture to start
    # ==========================
    if session.state == STATE_IDLE:
        if hand_visible:
            session.pre_roll.append(features)

        if hand_visible and session.detector.detect_motion(features):
            session.buffer.clear()
            session.prediction_history.clear()
            session.buffer.prepend(list(session.pre_roll))

            session.detector.reset()
            session.detector.begin_recording()
            session.state = STATE_RECORDING

        return "Waiting for gesture...", None

    # ==========================
    # RECORDING: capturing frames until the gesture finishes
    # ==========================
    session.buffer.add(features)

    gesture_ended = session.detector.detect_end(features)
    buffer_full = session.buffer.is_full()

    if not (session.buffer.is_ready() and (gesture_ended or buffer_full)):
        progress = min(session.buffer.size() / SEQUENCE_LENGTH, 1.0)
        return f"Recording... {int(progress * 100)}%", None

    # gesture complete - run prediction
    sequence = session.buffer.get_sequence()
    result = predictor.predict(sequence)
    sign = result["sign"]
    confidence = result["confidence"]

    text = "Unknown sign"
    if sign != "Unknown":
        session.prediction_history.append(sign)
        vote = Counter(session.prediction_history).most_common(1)

        if vote:
            final_sign = vote[0][0]
            text = f"{final_sign} ({confidence * 100:.1f}%)"

            can_speak = (
                final_sign != session.last_spoken
                or (time.time() - session.last_spoken_time) > SPEAK_COOLDOWN
            )
            if can_speak:
                try:
                    tts.speak(final_sign)
                except Exception as e:
                    print(f"TTS error: {e}")
                session.last_spoken = final_sign
                session.last_spoken_time = time.time()

    session.reset()
    return text, confidence


# -----------------------------
# Reset buffer
# -----------------------------
def reset():
    session.reset()
    return "Reset", None


# -----------------------------
# Gradio Interface
# -----------------------------
with gr.Blocks(title="AI Sign Language Translator") as demo:

    gr.Markdown(
        """
        # 🤟 AI Sign Language Translator

        Real-time sign language recognition using:

        - MediaPipe
        - LSTM Neural Network
        - Text To Speech

        Hold your hand still until "Waiting for gesture..." appears, then perform a sign.
        Recording starts automatically and stops once the sign is complete.
        """
    )

    with gr.Row():
        webcam = gr.Image(
            sources=["webcam"],
            streaming=True,
            type="numpy",
            label="Webcam"
        )

    output = gr.Textbox(label="Detected Sign")
    confidence = gr.Number(label="Confidence")

    reset_btn = gr.Button("Reset")

    webcam.stream(
        fn=predict_sign,
        inputs=webcam,
        outputs=[output, confidence]
    )

    reset_btn.click(
        fn=reset,
        outputs=[output, confidence]
    )


# HuggingFace entry
if __name__ == "__main__":
    demo.launch()