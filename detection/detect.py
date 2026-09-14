from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Detect plastic waste in an image
results = model.predict(
    source="detection",
    conf=0.25,
    save=True
)

# Display detection count
for result in results:
    print("Detected objects:", len(result.boxes))