from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolov8n.pt')  # This will auto-download on first run

# Train the model
results = model.train(
    data='datacenter_dataset/data.yaml',  # Path to your dataset
    epochs=100,
    imgsz=640,
    batch=4,  # Smaller batch for CPU training
    patience=10,  # Early stopping
    save=True,
    cache=True
)

print("Training complete!")
print(f"Best model saved to: runs/detect/train/weights/best.pt")