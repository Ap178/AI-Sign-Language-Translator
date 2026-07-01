# speech/tts.py
# Use:

# ```python
# pyttsx3
# ```

# or

# ```python
# gTTS
# ```

# Input:

# ```
# "Hello"
# ```

# Output:

# voice.

# ---

import os
import time
from gtts import gTTS
import platform


class TextToSpeech:


    def __init__(self):

        self.last_text = ""
        self.last_time = 0

        # prevent repeating same word too often
        self.cooldown = 2



    def speak(self, text):


        if text is None:
            return



        text = str(text)



        # ignore unknown

        if text.lower() == "unknown":
            return



        current_time = time.time()



        # avoid repeating

        if (
            text == self.last_text
            and current_time - self.last_time < self.cooldown
        ):
            return



        self.last_text = text
        self.last_time = current_time



        print(
            f"Speaking: {text}"
        )


        try:


            filename = "speech_output.mp3"



            tts = gTTS(
                text=text,
                lang="en",
                slow=False
            )


            tts.save(
                filename
            )


            self.play_audio(
                filename
            )


            # remove file

            if os.path.exists(filename):

                os.remove(filename)



        except Exception as e:

            print(
                "TTS Error:",
                e
            )




    def play_audio(self, filename):


        system = platform.system()



        try:


            if system == "Linux":

                os.system(
                    f"mpg123 {filename}"
                )


            elif system == "Windows":

                os.system(
                    f"start {filename}"
                )


            elif system == "Darwin":

                os.system(
                    f"afplay {filename}"
                )


        except Exception as e:

            print(
                "Audio Error:",
                e
            )





if __name__ == "__main__":


    tts = TextToSpeech()


    tts.speak(
        "Hello, welcome"
    )