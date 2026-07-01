# ```python
# detect_expression(frame)
# ```
import cv2
import numpy as np
import mediapipe as mp


from utils.config import (
    FACE_FEATURE_SIZE
)



class FaceDetector:


    def __init__(self):


        self.mp_face = (
            mp.solutions.face_mesh
        )


        self.face = (
            self.mp_face.FaceMesh(

                max_num_faces=1,

                refine_landmarks=True,

                min_detection_confidence=0.5,

                min_tracking_confidence=0.5

            )
        )



    def extract(self,frame):


        """
        Extract face landmarks

        Returns:
        1404 values
        """


        rgb=cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        result=self.face.process(
            rgb
        )


        features=[]



        if result.multi_face_landmarks:


            face=result.multi_face_landmarks[0]


            for point in face.landmark:


                features.extend(
                    [
                        point.x,
                        point.y,
                        point.z
                    ]
                )



        while len(features)<FACE_FEATURE_SIZE:

            features.append(0.0)



        return np.array(
            features[:FACE_FEATURE_SIZE],
            dtype=np.float32
        )