# inference/gesture_detector.py

import time
import numpy as np
from collections import deque


class GestureStartDetector:
    """
    Tracks motion between consecutive landmark frames to detect when a
    gesture starts and when it ends (hand becomes still again).

    Start detection reacts quickly (short confirmation window) so we don't
    lose the first frames of the sign. End detection is time-based, with a
    minimum hold duration, so brief mid-sign pauses don't end recording early.
    """

    def __init__(
        self,
        start_threshold=0.015,
        start_confirm_frames=2,       # frames of motion needed to confirm START (fast trigger)
        end_threshold=0.012,          # below this = "still" for end detection
        end_stillness_seconds=0.45,   # how long it must stay still to confirm END
        min_recording_seconds=0.6,    # don't even check for END before this much time has passed
    ):
        self.start_threshold = start_threshold
        self.start_confirm_frames = start_confirm_frames
        self.end_threshold = end_threshold
        self.end_stillness_seconds = end_stillness_seconds
        self.min_recording_seconds = min_recording_seconds

        self.prev_frame = None
        self.motion_history = deque(maxlen=start_confirm_frames)

        self.recording_start_time = None
        self.still_since = None

    # ---------------- START ----------------

    def detect_motion(self, frame):
        """
        Returns True once a short burst of movement is detected (gesture start).
        Fast to trigger by design - pair this with a pre-roll buffer in the
        caller so the frames just before the trigger aren't lost.
        """
        if frame is None:
            return False

        if self.prev_frame is None:
            self.prev_frame = frame
            return False

        motion = float(np.mean(np.abs(frame - self.prev_frame)))
        self.prev_frame = frame
        self.motion_history.append(motion)

        if len(self.motion_history) < self.start_confirm_frames:
            return False

        return np.mean(self.motion_history) > self.start_threshold

    # ---------------- END ----------------

    def begin_recording(self):
        """Call once, exactly when recording starts."""
        self.recording_start_time = time.time()
        self.still_since = None

    def detect_end(self, frame):
        """
        Call every frame while recording. Returns True once the hand has
        been still for `end_stillness_seconds`, AND at least
        `min_recording_seconds` has elapsed since begin_recording().
        """
        now = time.time()

        if self.recording_start_time is None:
            self.recording_start_time = now

        elapsed = now - self.recording_start_time
        if elapsed < self.min_recording_seconds:
            # too early to end, but still track motion for continuity
            if frame is not None:
                self.prev_frame = frame
            return False

        if frame is None:
            # no hand visible counts as "still" (hand likely dropped/finished)
            is_still = True
        else:
            if self.prev_frame is None:
                self.prev_frame = frame
                return False
            motion = float(np.mean(np.abs(frame - self.prev_frame)))
            self.prev_frame = frame
            is_still = motion < self.end_threshold

        if is_still:
            if self.still_since is None:
                self.still_since = now
            return (now - self.still_since) >= self.end_stillness_seconds
        else:
            self.still_since = None
            return False

    def reset(self):
        self.prev_frame = None
        self.motion_history.clear()
        self.recording_start_time = None
        self.still_since = None