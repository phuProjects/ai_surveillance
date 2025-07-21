import cv2
from ultralytics import YOLO
import simpleaudio as sa
import os


model = YOLO("yolov8n.pt")
alert_wave = sa.WaveObject.from_wave_file(os.path.join("alerts", "bloop_x.wav"))

TARGET_OBJECTS = {"bottle"} #Add more objects later on

def alert():
    alert_wave.play()

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

    #Checking for detections
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label in TARGET_OBJECTS:
                print(f"ALERT {label} detected!!!")
                alert()
                
    cv2.imshow("YOLOv8 Surveillance", annotated_frame)

    if cv2.waitKey(1) == 27:  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
