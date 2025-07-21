import cv2
from ultralytics import YOLO
import pygame as pg
import os
import time


model = YOLO("yolov8n.pt")
pg.mixer.init()

alert_path = os.path.join("alerts","pop.wav")
alert_sound = pg.mixer.Sound(alert_path)

TARGET_OBJECTS = {"bottle"} #Add more objects later on
ALERT_COOLDOWN = 5   # seconds
last_alert_time = 0  # track time of last alert

def alert():
    global last_alert_time
    now = time.time()
    if now - last_alert_time >= ALERT_COOLDOWN:
        try:
            alert_sound.play()
            last_alert_time = now
        except Exception as e:
            print("Audio playback failed:", e)

video_path = os.path.join("videos","")
cap = cv2.VideoCapture(video_path)
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
