from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Detect plastic waste
results = model.predict(
    source="detection",
    conf=0.25
)

# Calculate hotspot level for each image
for result in results:
    count = len(result.boxes)

    if count == 0:
        level = "Low"
    elif count <= 2:
        level = "Medium"
    else:
        level = "High"

    print("Image:", result.path)
    print("Plastic waste detected:", count)
    print("Waste level:", level)
    print("----------------------")