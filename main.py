import cv2
from ultralytics import YOLO


model = YOLO("yolov8n.pt")

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

    if cv2.waitKey(100) == 27:  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
