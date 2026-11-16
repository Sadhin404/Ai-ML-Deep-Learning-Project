from ultralytics import YOLO
import cv2

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model =YOLO("yolov8m.pt") #using model v8 v7 slow kaj korceh 

video = "traffic.mp4" 
framelim =5400 
labels =True #for lebel

cap = cv2.VideoCapture(video)
if not cap.isOpened():
    print("VIdeo loading error")
    exit() 

fps = cap.get(cv2.CAP_PROP_FPS)

frame_count =0

while True:
    ret, frame = cap.read()
    if not ret or frame_count >framelim:
        break
    frame_count += 1

    results = model(frame,verbose=False)

    for r in results: #for box 
        for box in r.boxes:
            cls_id =int(box.cls[0])

            label = model.names[cls_id]
            conf = float(box.conf[0])

            if label in ["car", "truck", "bus","motorbike"]:
                x1, y1, x2, y2 =map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if labels:
                    text = f"{label} {conf:.2f}"
                    cv2.putText(frame, text, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 2)

    cv2.imshow("YOLO Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()

print("Finished")
