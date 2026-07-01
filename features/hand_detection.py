# Functions:

# ```python
# extract_hand_landmarks(frame)
# ```

# Output:

# ```python
# (126,)
# ```

# Example:

# ```python
# [
# x1,y1,z1,
# x2,y2,z2,
# ...
# ]
# ```
import cv2
import numpy as np
import mediapipe as mp

from utils.config import (
    MAX_HANDS,
    HAND_FEATURE_SIZE
)



class HandDetector:


    def __init__(self):


        self.mp_hands = (
            mp.solutions.hands
        )


        self.hands = (
            self.mp_hands.Hands(

                static_image_mode=False,

                max_num_hands=MAX_HANDS,

                min_detection_confidence=0.5,

                min_tracking_confidence=0.5
            )
        )



    def extract(self, frame):


        """
        Input:
            BGR frame

        Output:
            126 features
        """


        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        result = self.hands.process(
            rgb
        )


        landmarks=[]



        if result.multi_hand_landmarks:


            for hand in result.multi_hand_landmarks:


                for point in hand.landmark:


                    landmarks.extend(
                        [
                            point.x,
                            point.y,
                            point.z
                        ]
                    )



        # padding

        while len(landmarks) < HAND_FEATURE_SIZE:

            landmarks.append(0.0)



        return np.array(
            landmarks[:HAND_FEATURE_SIZE],
            dtype=np.float32
        )



    def draw(self,frame):


        rgb=cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        result=self.hands.process(rgb)



        if result.multi_hand_landmarks:


            for hand in result.multi_hand_landmarks:


                self.mp_hands.drawing_utils.draw_landmarks(

                    frame,

                    hand,

                    self.mp_hands.HAND_CONNECTIONS
                )


        return frame