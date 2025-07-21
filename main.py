import cv2
from ultralytics import YOLO
from playsound import playsound
import os


model = YOLO("yolov8n.pt")

ALERT_SOUND = os.path.join("alrts", "bloop_x.wav") #Using os make program compatible with other operating systems
TARGET_OBJECTS = {"bottle"} #Add more objects later on

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

    #Checking for detections
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label in TARGET_OBJECTS:
                print(f"ALERT {label} detected!!!")

                if not alert_triggered:
                    playsound(ALERT_SOUND)
                    alert_triggered = True
                    
    alert_triggered = False


    cv2.imshow("YOLOv8 Surveillance", annotated_frame)

    if cv2.waitKey(1) == 27:  # Press 'q' to quit
        break


cap.release()
cv2.destroyAllWindows()
