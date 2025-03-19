import cv2
import torch
from ultralytics import YOLO
import pyttsx3
import time
import threading
from vosk import Model, KaldiRecognizer
import pyaudio
import json
from obj_detection import detect_object
from live import object_detection
from audio import speech_to_text
from queue import Queue
from fuzzywuzzy import process
import nltk
from nltk.corpus import process

queue = Queue()

# Create threads for speech-to-text and object detection
speech_thread = threading.Thread(target=speech_to_text, args=(queue,))
detection_thread = threading.Thread(target=object_detection, args=(queue,))

# Start both threads
speech_thread.start()
detection_thread.start()

# Wait for threads to finish
speech_thread.join()
detection_thread.join()