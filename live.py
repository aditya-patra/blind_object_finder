import cv2
import torch
from ultralytics import YOLO
import pyttsx3
import time
from queue import Queue
from fuzzywuzzy import process
from word_match import get_synonym_match, hard_to_locate_objects, yolo_classes

def object_detection(queue):
    # Load the YOLOv8 model
    model = YOLO("yolov8n.pt")  # Use "yolov8s.pt" for a slightly better model
    # Initialize the TTS engine
    engine = pyttsx3.init()

    # Set properties (optional)
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

    # Open the webcam (0 for the default camera)
    cap = cv2.VideoCapture(0)

    # Set the frame width and height (optional)
    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height
    prev_queue = ""
    curr_queue = ""
    first_word = False
    while cap.isOpened():
        if queue.empty():
            curr_queue = prev_queue
        else:
            curr_queue = queue.get()
            closest_match = process.extractOne(curr_queue, yolo_classes)
            curr_queue = closest_match
        if curr_queue != prev_queue and curr_queue != "":
            prev_queue = curr_queue
            engine.say("Searching for "+curr_queue)
            # Wait for completion
            engine.runAndWait()
        if(curr_queue == "" and prev_queue == ""):
            continue
        if curr_queue == "end" or curr_queue == "stop" or curr_queue == "exit":
            continue
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Draw the detected objects on the frame
        for result in results:
            text=""
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = box.conf[0].item()  # Confidence score
                cls = int(box.cls[0].item())  # Class index
                label = f"{model.names[cls]} {conf:.2f}"
                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(label)
                if curr_queue in label:
                    x_center, y_center, width, height = box.xywh[0]
                    if not(640 / 2 - 20 < x_center < 640 / 2 + 20):
                        if x_center < 640/2-20:
                            text = "Move left"
                        else:
                            text = "Move right"
                    elif not(480 / 2 - 20 < y_center < 480 / 2 + 20):
                        if y_center < 480/2-20:
                            text = "Move up"
                        else:
                            text = "Move down"
                    else:
                        text = "Centered"
            engine.say(text)

            # Wait for completion
            engine.runAndWait()

        # Show the frame
        cv2.imshow("YOLOv8 Live Object Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()