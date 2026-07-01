# Purpose:

# Increase training data.

# Adds:

# noise
# scaling
# small movement variation

import os
import numpy as np

from tqdm import tqdm

from utils.config import LANDMARK_DIR
from utils.helpers import create_directory





def add_noise(sequence):


    noise=np.random.normal(

        0,

        0.005,

        sequence.shape

    )


    return sequence + noise





def scale(sequence):


    factor=np.random.uniform(
        0.9,
        1.1
    )


    return sequence * factor






def augment_dataset():


    gestures=os.listdir(
        LANDMARK_DIR
    )



    for gesture in gestures:


        folder=os.path.join(
            LANDMARK_DIR,
            gesture
        )


        files=os.listdir(
            folder
        )


        for file in tqdm(
            files,
            desc=gesture
        ):


            path=os.path.join(
                folder,
                file
            )


            data=np.load(
                path
            )



            # noisy version

            noisy=add_noise(
                data
            )


            np.save(

                os.path.join(
                    folder,
                    file.replace(
                        ".npy",
                        "_noise.npy"
                    )
                ),

                noisy
            )



            # scaled version

            scaled=scale(
                data
            )


            np.save(

                os.path.join(
                    folder,
                    file.replace(
                        ".npy",
                        "_scale.npy"
                    )
                ),

                scaled
            )





if __name__=="__main__":

    augment_dataset()