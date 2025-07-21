import cv2
from ultralytics import YOLO
from playsound import playsound
import os


model = YOLO("yolov8n.pt")

ALERT_SOUND = os.path.join("alrts", "bloop_x.wav")
TARGET_OBJECTS = {"person"}

alert_triggered = False

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Surveillance", annotated_frame)

    if cv2.waitKey(1) == 27:  # Press 'q' to quit
        break


cap.release()
cv2.destroyAllWindows()
