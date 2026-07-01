# Purpose:

# Convert:

# video
#  |
# MediaPipe
#  |
# 30 frames
#  |
# 1530 features
#  |
# .npy

import cv2
import os
import numpy as np
from tqdm import tqdm


from features.feature_fusion import FeatureExtractor

from utils.config import (
    RAW_VIDEO_DIR,
    LANDMARK_DIR,
    SEQUENCE_LENGTH
)

from utils.helpers import create_directory



extractor = FeatureExtractor()



def process_video(video_path):


    cap=cv2.VideoCapture(
        video_path
    )


    frames=[]


    while True:


        ret,frame=cap.read()


        if not ret:
            break


        frames.append(frame)



    cap.release()



    if len(frames)==0:

        return None



    # choose 30 frames

    indices=np.linspace(

        0,

        len(frames)-1,

        SEQUENCE_LENGTH

    ).astype(int)



    sequence=[]



    for i in indices:


        feature = extractor.extract(
            frames[i]
        )


        sequence.append(
            feature
        )



    return np.array(
        sequence,
        dtype=np.float32
    )






def extract_all():


    create_directory(
        LANDMARK_DIR
    )


    gestures=os.listdir(
        RAW_VIDEO_DIR
    )



    for gesture in gestures:


        input_dir=os.path.join(
            RAW_VIDEO_DIR,
            gesture
        )


        output_dir=os.path.join(
            LANDMARK_DIR,
            gesture
        )


        create_directory(
            output_dir
        )



        videos=os.listdir(
            input_dir
        )



        for video in tqdm(
            videos,
            desc=gesture
        ):


            video_path=os.path.join(
                input_dir,
                video
            )


            output_path=os.path.join(
                output_dir,
                video.replace(
                    ".mp4",
                    ".npy"
                )
            )



            sequence=process_video(
                video_path
            )



            if sequence is not None:


                np.save(
                    output_path,
                    sequence
                )




if __name__=="__main__":

    extract_all()