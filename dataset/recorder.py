# Purpose:

# Record gesture videos
# Automatically save them
# Organize by gesture name
import cv2
import os
import time

from utils.config import RAW_VIDEO_DIR
from utils.helpers import create_directory



# Recording settings

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

FPS = 30

RECORD_SECONDS = 3



def countdown(cap):

    for i in range(3,0,-1):

        ret, frame = cap.read()

        if ret:

            cv2.putText(
                frame,
                str(i),
                (300,250),
                cv2.FONT_HERSHEY_SIMPLEX,
                5,
                (0,255,0),
                5
            )

            cv2.imshow(
                "Recorder",
                frame
            )

            cv2.waitKey(1000)




def record_video(
        gesture,
        sample_id
):


    save_dir = os.path.join(
        RAW_VIDEO_DIR,
        gesture
    )


    create_directory(
        save_dir
    )


    path = os.path.join(
        save_dir,
        f"{sample_id}.mp4"
    )


    cap = cv2.VideoCapture(0)


    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )


    print("\nPrepare...")


    time.sleep(2)


    countdown(cap)



    print("Recording...")


    writer = cv2.VideoWriter(

        path,

        cv2.VideoWriter_fourcc(*"mp4v"),

        FPS,

        (
            FRAME_WIDTH,
            FRAME_HEIGHT
        )

    )



    start=time.time()


    while True:


        ret,frame = cap.read()


        if not ret:
            break



        writer.write(frame)



        cv2.putText(
            frame,
            gesture,
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )


        cv2.imshow(
            "Recorder",
            frame
        )


        if time.time()-start >= RECORD_SECONDS:

            break



        if cv2.waitKey(1)==27:

            break



    writer.release()

    cap.release()

    cv2.destroyAllWindows()



    print(
        f"Saved: {path}"
    )





if __name__=="__main__":


    gesture=input(
        "Enter gesture name: "
    )


    count=int(
        input(
            "Number of samples: "
        )
    )



    for i in range(count):

        record_video(
            gesture,
            i+1
        )