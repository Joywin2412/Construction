import yaml
from ultralytics import YOLO
import cv2
model = YOLO('best.pt')

results = model.predict(cv2.imread('test/images/00000009_jpg.rf.f10e6c49ae2a7b150dda70d1ff04dd8b.jpg'))
result = results[0]
output = []
for box in result.boxes:
    x1, y1, x2, y2 = [
        round(x) for x in box.xyxy[0].tolist()
    ]
    class_id = box.cls[0].item()
    prob = round(box.conf[0].item(), 2)
   
    print(x1, y1, x2, y2, result.names[class_id], prob)
    