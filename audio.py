from vosk import Model, KaldiRecognizer
import pyaudio
import json
from queue import Queue

def speech_to_text(queue):
    # Load the Vosk model
    model = Model("vosk-model-small-en-us-0.15")

    # Initialize recognizer
    recognizer = KaldiRecognizer(model, 16000)

    # Set up the microphone stream
    mic = pyaudio.PyAudio()
    stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
    stream.start_stream()

    print("Listening...")

    # Process audio data in real-time
    while True:
        data = stream.read(4000, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            if result["text"] != "":
                print("You said:", result["text"])
                queue.put(result["text"])  # Put the recognized word in the queue