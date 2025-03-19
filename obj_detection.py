import cv2
from ultralytics import YOLO
from queue import Queue

def detect_object(queue):
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')  # You can use 'yolov8s.pt', 'yolov8m.pt', etc., for different model sizes

    # Load the image
    image_path = '000000000009.jpg'
    image = cv2.imread(image_path)

    # Perform object detection
    results = model(image)

    # Display the results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class ID

            # Draw the bounding box and label
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the image
    cv2.imshow("Detection Results", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()