import cv2
import os
import numpy as np
import time

from features.feature_fusion import FeatureExtractor



SAVE_DIR = "dataset/landmarks"



class SignCollector:


    def __init__(self):

        self.extractor = FeatureExtractor()


    def collect(
        self,
        sign_name,
        samples=50
    ):


        folder = os.path.join(
            SAVE_DIR,
            sign_name
        )

        os.makedirs(
            folder,
            exist_ok=True
        )


        cap = cv2.VideoCapture(0)


        print(
            f"Collecting {sign_name}"
        )


        count = 0



        while count < samples:


            ret, frame = cap.read()


            if not ret:
                break


            frame = cv2.flip(
                frame,
                1
            )


            cv2.putText(
                frame,
                f"Sample {count+1}/{samples}",
                (30,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )



            cv2.imshow(
                "Collect Sign",
                frame
            )



            key = cv2.waitKey(1)



            if key == ord('s'):


                sequence=[]


                print(
                    "Recording..."
                )


                for i in range(30):


                    ret, frame = cap.read()


                    frame = cv2.flip(
                        frame,
                        1
                    )


                    landmarks = (
                        self.extractor.extract(
                            frame
                        )
                    )


                    sequence.append(
                        landmarks
                    )


                    time.sleep(
                        0.03
                    )



                sequence=np.array(
                    sequence
                )



                np.save(
                    os.path.join(
                        folder,
                        f"{count}.npy"
                    ),
                    sequence
                )


                count += 1


                print(
                    "saved",
                    count
                )



            if key == ord('q'):
                break



        cap.release()

        cv2.destroyAllWindows()




if __name__=="__main__":


    sign=input(
        "Enter new sign name: "
    )


    collector=SignCollector()


    collector.collect(
        sign,
        samples=50
    )