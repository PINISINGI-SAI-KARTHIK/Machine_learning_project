from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # small, fast pretrained model

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="connector_detector"
)
