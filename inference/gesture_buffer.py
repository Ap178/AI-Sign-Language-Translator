# inference/gesture_buffer.py

from collections import deque
import numpy as np


class GestureBuffer:
    """
    Holds a rolling sequence of per-frame landmark features for one gesture.
    """

    def __init__(self, sequence_length=30, min_frames=15):
        self.sequence_length = sequence_length  # LSTM input length
        self.min_frames = min_frames            # earliest point a prediction is allowed
        self.buffer = deque(maxlen=sequence_length)

    def add(self, landmarks):
        """Add one frame's landmark feature vector."""
        if landmarks is not None:
            self.buffer.append(landmarks)

    def prepend(self, frames):
        """
        Insert a list of earlier frames (e.g. a pre-roll buffer captured
        just before the gesture was officially detected) at the front of
        the sequence, without dropping anything already in the buffer.
        """
        if not frames:
            return
        existing = list(self.buffer)
        combined = list(frames) + existing
        combined = combined[-self.sequence_length:]
        self.buffer = deque(combined, maxlen=self.sequence_length)

    def is_ready(self):
        """True once we have enough frames to attempt a prediction."""
        return len(self.buffer) >= self.min_frames

    def is_full(self):
        """True once the buffer has reached the full sequence length."""
        return len(self.buffer) >= self.sequence_length

    def get_sequence(self):
        """
        Returns a (sequence_length, feature_dim) array, padding the start
        with the earliest frame if fewer than sequence_length were captured.
        """
        if not self.is_ready():
            return None

        data = list(self.buffer)

        if len(data) < self.sequence_length:
            padding = [data[0]] * (self.sequence_length - len(data))
            data = padding + data

        return np.array(data, dtype=np.float32)

    def clear(self):
        self.buffer.clear()

    def size(self):
        return len(self.buffer)