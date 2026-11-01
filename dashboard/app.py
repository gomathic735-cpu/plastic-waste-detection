import streamlit as st
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
import tempfile
import os
import csv

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Plastic Waste Hotspot Dashboard",
    page_icon="♻️",
    layout="wide"
)

st.title("AI-Based Plastic Waste Detection and Hotspot Mapping")

st.write(
    "Upload a shoreline image to detect plastic waste "
    "and calculate the prototype hotspot score."
)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "train",
    "weights",
    "best.pt"
)

HOTSPOT_FOLDER = os.path.join(BASE_DIR, "hotspot")
os.makedirs(HOTSPOT_FOLDER, exist_ok=True)

CSV_PATH = os.path.join(HOTSPOT_FOLDER, "locations.csv")

CSV_HEADERS = [
    "image",
    "section",
    "plastic_count",
    "plastic_area",
    "hotspot_score",
    "hotspot_level"
]

# --------------------------------------------------
# CREATE CSV FILE
# --------------------------------------------------

try:
    if (
        not os.path.exists(CSV_PATH)
        or os.path.getsize(CSV_PATH) == 0
    ):
        with open(
            CSV_PATH,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(CSV_HEADERS)

except OSError as error:
    st.error(f"Unable to create CSV file: {error}")
    st.stop()

# --------------------------------------------------
# LOAD YOLO11n MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


try:
    if not os.path.exists(MODEL_PATH):
        st.error("Trained YOLO11n model file was not found.")
        st.stop()

    model = load_model()

except Exception as error:
    st.error(f"Unable to load YOLO model: {error}")
    st.stop()

# --------------------------------------------------
# SELECT SHORELINE SECTION
# --------------------------------------------------

section = st.selectbox(
    "Select Lake Shoreline Section",
    [
        "Section A",
        "Section B",
        "Section C",
        "Section D"
    ]
)

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a shoreline image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:

    st.info("Please upload a shoreline image to begin detection.")

else:

    image_path = None

    try:

        # Validate uploaded image
        image = Image.open(uploaded_file)
        image.verify()

        # Reopen image after verification
        uploaded_file.seek(0)

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Shoreline Image",
            width="stretch"
        )

        # --------------------------------------------------
        # SAVE TEMPORARY IMAGE
        # --------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as temp_file:

            image.save(temp_file.name)
            image_path = temp_file.name

        # --------------------------------------------------
        # YOLO11n DETECTION
        # --------------------------------------------------

        st.subheader("AI Detection Result")

        with st.spinner("Detecting plastic waste..."):

            results = model.predict(
                source=image_path,
                conf=0.25
            )

        result = results[0]

        detected_image = result.plot()

        st.image(
            detected_image,
            caption="YOLO11n Plastic Waste Detection",
            width="stretch"
        )

        # --------------------------------------------------
        # PLASTIC COUNT
        # --------------------------------------------------

        count = len(result.boxes)

        # --------------------------------------------------
        # CALCULATE BOUNDING BOX AREA
        # --------------------------------------------------

        total_area = 0

        for box in result.boxes.xyxy:

            x1, y1, x2, y2 = box

            width = x2 - x1
            height = y2 - y1

            total_area += width * height

        total_area = float(total_area)

        # --------------------------------------------------
        # HOTSPOT SCORE
        # --------------------------------------------------

        score = count * 10 + (total_area / 10000)

        if score == 0:
            level = "Low"

        elif score < 50:
            level = "Medium"

        else:
            level = "High"

        # --------------------------------------------------
        # DISPLAY RESULTS
        # --------------------------------------------------

        st.subheader("AI Detection Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Plastic Objects", count)

        with col2:
            st.metric(
                "Plastic Area",
                f"{total_area:.2f} pixels"
            )

        with col3:
            st.metric(
                "Hotspot Score",
                f"{score:.2f}"
            )

        st.write(f"**Shoreline Section:** {section}")

        st.write(f"**Hotspot Level:** {level}")

        # --------------------------------------------------
        # SAVE RESULTS TO CSV
        # --------------------------------------------------

        if "saved_records" not in st.session_state:
            st.session_state.saved_records = set()

        record_key = (
            uploaded_file.name,
            section,
            round(score, 2)
        )

        if record_key not in st.session_state.saved_records:

            try:
                with open(
                    CSV_PATH,
                    "a",
                    newline="",
                    encoding="utf-8"
                ) as file:

                    writer = csv.writer(file)

                    writer.writerow([
                        uploaded_file.name,
                        section,
                        count,
                        round(total_area, 2),
                        round(score, 2),
                        level
                    ])

                st.session_state.saved_records.add(record_key)

                st.success("Hotspot result saved successfully!")

            except OSError as error:

                st.error(f"Unable to save hotspot record: {error}")

        # --------------------------------------------------
        # HOTSPOT LEVEL MESSAGE
        # --------------------------------------------------

        st.subheader("Hotspot Level")

        if level == "Low":

            st.success("Low plastic waste detected.")

        elif level == "Medium":

            st.warning("Medium plastic waste detected.")

        else:

            st.error("High hotspot score detected.")

    except (UnidentifiedImageError, OSError, ValueError):

        st.error(
            "Invalid or corrupted image. "
            "Please upload a valid JPG, JPEG or PNG image."
        )

    except Exception as error:

        st.error(f"Error during detection: {error}")

    finally:

        if image_path and os.path.exists(image_path):
            os.remove(image_path)

# --------------------------------------------------
# SECTION-WISE COMPARISON USING CSV HISTORY
# --------------------------------------------------

st.divider()

st.subheader("Section-wise Hotspot Comparison")

sections = [
    "Section A",
    "Section B",
    "Section C",
    "Section D"
]

section_totals = {
    name: 0 for name in sections
}

section_counts = {
    name: 0 for name in sections
}

try:

    with open(
        CSV_PATH,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        records = csv.DictReader(file)

        for record in records:

            name = record["section"]

            score_value = float(record["hotspot_score"])

            if name in section_totals:

                section_totals[name] += score_value
                section_counts[name] += 1

    average_scores = {}

    for name in sections:

        if section_counts[name] > 0:

            average_scores[name] = (
                section_totals[name] / section_counts[name]
            )

        else:

            average_scores[name] = 0

    for name, average in average_scores.items():

        st.write(
            f"**{name}:** {average:.2f} "
            f"({section_counts[name]} records)"
        )

    st.subheader("Hotspot Score Comparison")

    st.bar_chart(average_scores)

except (OSError, ValueError, KeyError) as error:

    st.error(f"Unable to compare hotspot records: {error}")

# --------------------------------------------------
# DISPLAY SAVED RECORDS
# --------------------------------------------------

st.divider()

st.subheader("Saved Hotspot Records")

try:

    with open(
        CSV_PATH,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        saved_records = list(csv.DictReader(file))

    if saved_records:

        st.dataframe(
            saved_records,
            width="stretch"
        )

        st.success(
            f"Total saved records: {len(saved_records)}"
        )

    else:

        st.info("No hotspot records available yet.")

except (OSError, csv.Error) as error:

    st.error(f"Unable to read saved records: {error}")

# --------------------------------------------------
# PROJECT NOTE
# --------------------------------------------------

st.caption(
    "Note: Hotspot scores are prototype values based on image pixel area. "
    "They are not real-world waste measurements or GPS-based locations."
)