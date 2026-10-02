from ultralytics import YOLO

try:
    # Load trained YOLO11n model
    model = YOLO("runs/detect/train/weights/best.pt")

    # Detect plastic waste in field images
    results = model.predict(
        source="detection",
        conf=0.25
    )

    # Analyse each image
    for result in results:

        # Number of detected plastic objects
        count = len(result.boxes)

        # Calculate total detected plastic area
        total_area = 0

        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = box

            width = x2 - x1
            height = y2 - y1

            area = width * height
            total_area = total_area + area

        # Calculate prototype hotspot score
        score = count * 10 + (float(total_area) / 10000)

        # Determine hotspot level
        if score == 0:
            level = "Low"
        elif score < 50:
            level = "Medium"
        else:
            level = "High"

        print("Image:", result.path)
        print("Plastic waste detected:", count)
        print("Detected plastic area:", round(float(total_area), 2), "pixels")
        print("Hotspot score:", round(score, 2))
        print("Hotspot level:", level)
        print("----------------------")

except FileNotFoundError:
    print("Error: Required model or file was not found.")

except Exception as error:
    print("Error while running hotspot analysis:", error)