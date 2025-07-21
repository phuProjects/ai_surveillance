import cv2
from ultralytics import YOLO
import simpleaudio as sa
import os
import time


model = YOLO("yolov8n.pt")

ALERT_SOUND_PATH = os.path.join("alerts", "pop.wav") #Using os make program compatible with other operating systems
alert_wave = sa.WaveObject.from_wave_file(ALERT_SOUND_PATH)
TARGET_OBJECTS = {"bottle"} #Add more objects later on

last_alert_time = 0  # track time of last alert
ALERT_COOLDOWN = 5   # seconds

def alert():
    global last_alert_time
    now = time.time()
    if now - last_alert_time >= ALERT_COOLDOWN:
        try:
            alert_wave.play()
            last_alert_time = now
        except Exception as e:
            print("Audio playback failed:", e)



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
