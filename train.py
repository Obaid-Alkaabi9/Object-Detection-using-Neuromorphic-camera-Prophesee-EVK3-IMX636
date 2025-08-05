from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Train with early stopping by setting patience directly
model.train(
    data='data.yaml',
    epochs=500,
    imgsz=1280,
    patience=50  # 👈 Early stopping after 10 no-improvement epochs
)

