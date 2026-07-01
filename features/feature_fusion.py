# Combine:

# Hands:

# ```
# 126
# ```

# Face:

# ```
# optional
# ```

# Output:

# ```
# final feature vector
# ```

import numpy as np


from features.hand_detection import (
    HandDetector
)


from features.face_detection import (
    FaceDetector
)



class FeatureExtractor:


    def __init__(self):

        self.hand = HandDetector()

        self.face = FaceDetector()



    def extract(self,frame):


        hand_features = (
            self.hand.extract(frame)
        )


        face_features = (
            self.face.extract(frame)
        )


        combined = np.concatenate(

            [
                hand_features,
                face_features
            ]

        )


        return combined



    def draw(self,frame):


        frame = self.hand.draw(
            frame
        )


        return frame