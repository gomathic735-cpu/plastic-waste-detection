import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(
    page_title="Plastic Waste Hotspot Dashboard",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ AI-Based Plastic Waste Detection and Hotspot Mapping")

st.write(
    "Upload a shoreline image to detect plastic waste "
    "and calculate the prototype hotspot score."
)

# Load trained YOLO11n model
model = YOLO("runs/detect/train/weights/best.pt")

# Select shoreline section
section = st.selectbox(
    "📍 Select Lake Shoreline Section",
    [
        "Section A",
        "Section B",
        "Section C",
        "Section D"
    ]
)

# Upload image
uploaded_file = st.file_uploader(
    "Upload a shoreline image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as temp_file:
        image.save(temp_file.name)
        image_path = temp_file.name

    # Run YOLO11n detection
    results = model.predict(
        source=image_path,
        conf=0.25
    )

    result = results[0]

    # Create image with AI bounding boxes
    detected_image = result.plot()

    st.subheader("🤖 AI Detection Result")

    st.image(
        detected_image,
        caption="YOLO11n Plastic Waste Detection",
        width="stretch"
    )

    # Count detected objects
    count = len(result.boxes)

    # Calculate total bounding-box area
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

    # Display results
    st.subheader("📊 AI Detection Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Plastic Objects",
            count
        )

    with col2:
        st.metric(
            "Plastic Area",
            f"{float(total_area):.2f} pixels"
        )

    with col3:
        st.metric(
            "Hotspot Score",
            f"{score:.2f}"
        )

    # Display selected shoreline section
    st.write(
        f"📍 **Shoreline Section:** {section}"
    )

    # Store section scores
    if "section_scores" not in st.session_state:
        st.session_state.section_scores = {
            "Section A": 0,
            "Section B": 0,
            "Section C": 0,
            "Section D": 0
        }

    st.session_state.section_scores[section] = score

    # Section-wise hotspot comparison
    st.subheader("📊 Section-wise Hotspot Comparison")

    for name in st.session_state.section_scores:
        st.write(
            f"📍 {name}: "
            f"{st.session_state.section_scores[name]:.2f}"
        )

    # Visual comparison
    st.subheader("📈 Hotspot Score Comparison")

    st.bar_chart(
        st.session_state.section_scores
    )

    # Hotspot level
    st.subheader("📍 Hotspot Level")

    st.write(
        f"**{level}**"
    )

    if level == "Low":
        st.success(
            "Low plastic waste detected."
        )

    elif level == "Medium":
        st.warning(
            "Medium plastic waste detected."
        )

    else:
        st.error(
            "High hotspot score detected."
        )

    # Remove temporary image
    os.remove(image_path)