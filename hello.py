# from ultralytics import YOLO

# model = YOLO('yolov8n.pt')
# model.predict(
#    source='https://media.roboflow.com/notebooks/examples/dog.jpeg',
#    conf=0.25
# )

from ultralytics import YOLO


model = YOLO('yolov8n.pt')
 
# Training.
results = model.train(
   data='C:/Users/hp/Documents/roboflow_dataset/data.yaml',
   imgsz=640,
   epochs=1,
   batch=8,
   name='yolov8n_custom')