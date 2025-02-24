from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.pt")
model = YOLO("runs/detect/train34/weights/best.pt")  # load a custom model
# model = YOLO("runs/detect/train34/weights/best.pt")  # load a custom model

# Customize validation settings
validation_results = model.val(data="hospital.yaml",conf=0.001, imgsz=512, device="0",max_det=5)
