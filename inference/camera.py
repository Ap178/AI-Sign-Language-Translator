# inference/camera.py

import cv2
import time

from collections import deque, Counter

from features.feature_fusion import FeatureExtractor
from inference.gesture_buffer import GestureBuffer
from inference.predictor import SignPredictor
from inference.gesture_detector import GestureStartDetector

from speech.tts import TextToSpeech


# ---- tunables ----
SEQUENCE_LENGTH = 30
MIN_FRAMES = 15
PRE_ROLL_FRAMES = 10       # frames kept in a rolling pre-buffer, prepended once recording starts
DISPLAY_DURATION = 2.0     # seconds the last prediction stays on screen
SPEAK_COOLDOWN = 1.5       # seconds before the same sign can be spoken again
VOTE_HISTORY = 5

STATE_IDLE = "idle"
STATE_RECORDING = "recording"


def draw_text(frame, text, pos, color, scale=1.0, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def main():
    extractor = FeatureExtractor()
    predictor = SignPredictor()
    tts = TextToSpeech()

    buffer = GestureBuffer(sequence_length=SEQUENCE_LENGTH, min_frames=MIN_FRAMES)
    detector = GestureStartDetector()

    # always-on rolling window of recent frames, used to "rescue" the start
    # of a gesture that happened just before official detection triggered
    pre_roll = deque(maxlen=PRE_ROLL_FRAMES)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    state = STATE_IDLE
    prediction_history = deque(maxlen=VOTE_HISTORY)

    last_prediction_text = ""
    last_prediction_time = 0.0
    last_spoken = ""
    last_spoken_time = 0.0

    prev_tick = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera frame not received, stopping.")
            break

        frame = cv2.flip(frame, 1)

        now = time.time()
        fps = 1.0 / max(now - prev_tick, 1e-6)
        prev_tick = now

        features = extractor.extract(frame)
        hand_visible = features is not None

        # ==========================
        # IDLE: waiting for a gesture to start
        # ==========================
        if state == STATE_IDLE:
            if hand_visible:
                pre_roll.append(features)

            if hand_visible and detector.detect_motion(features):
                print("Gesture started")

                buffer.clear()
                prediction_history.clear()

                # rescue the frames captured just before the trigger so we
                # don't lose the beginning of the sign
                buffer.prepend(list(pre_roll))

                detector.reset()
                detector.begin_recording()
                state = STATE_RECORDING

            draw_text(frame, "Waiting for gesture...", (30, 50), (0, 255, 255))

        # ==========================
        # RECORDING: capturing frames until the gesture finishes
        # ==========================
        else:
            buffer.add(features)
            draw_text(frame, "Recording", (30, 50), (0, 0, 255))

            progress = min(buffer.size() / SEQUENCE_LENGTH, 1.0)
            bar_w = 200
            cv2.rectangle(frame, (30, 65), (30 + bar_w, 80), (60, 60, 60), -1)
            cv2.rectangle(frame, (30, 65), (30 + int(bar_w * progress), 80), (0, 0, 255), -1)

            gesture_ended = detector.detect_end(features)
            buffer_full = buffer.is_full()

            if buffer.is_ready() and (gesture_ended or buffer_full):
                sequence = buffer.get_sequence()
                result = predictor.predict(sequence)
                sign = result["sign"]
                conf = result["confidence"]

                if sign != "Unknown":
                    prediction_history.append(sign)
                    vote = Counter(prediction_history).most_common(1)

                    if vote:
                        final_sign = vote[0][0]
                        last_prediction_text = f"{final_sign} ({conf * 100:.1f}%)"
                        last_prediction_time = time.time()

                        can_speak = (
                            final_sign != last_spoken
                            or (time.time() - last_spoken_time) > SPEAK_COOLDOWN
                        )
                        if can_speak:
                            tts.speak(final_sign)
                            last_spoken = final_sign
                            last_spoken_time = time.time()
                else:
                    last_prediction_text = "Unknown sign"
                    last_prediction_time = time.time()

                buffer.clear()
                pre_roll.clear()
                detector.reset()
                state = STATE_IDLE

        # ==========================
        # Overlays
        # ==========================
        if not hand_visible:
            draw_text(frame, "No hand detected", (30, 140), (0, 165, 255), scale=0.7)

        if time.time() - last_prediction_time < DISPLAY_DURATION and last_prediction_text:
            draw_text(frame, last_prediction_text, (30, 110), (0, 255, 0))

        draw_text(frame, f"FPS: {fps:.0f}", (30, frame.shape[0] - 20), (200, 200, 200), scale=0.6, thickness=1)

        cv2.imshow("AI Sign Language Translator", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()