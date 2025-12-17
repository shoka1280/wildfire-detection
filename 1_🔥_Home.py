import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import streamlit as st
import cv2
import requests
from PIL import Image
from glob import glob
from numpy import random
import io
from ultralytics import YOLO

# =========================
# Model Utilities
# =========================
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)


def predict_image(model, image, conf_threshold, iou_threshold):
    results = model.predict(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        device="cpu",
    )

    class_names = model.model.names
    classes = results[0].boxes.cls
    counts = {}

    for c in classes:
        name = class_names[int(c)]
        counts[name] = counts.get(name, 0) + 1

    if not counts:
        summary = "No wildfire-related objects detected."
    else:
        summary = "Detected " + ", ".join(
            f"{v} {k}{'s' if v > 1 else ''}" for k, v in counts.items()
        ) + "."

    img = results[0].plot()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img, summary


# =========================
# Main App
# =========================
def main():
    st.set_page_config(
        page_title="Wildfire Detection",
        page_icon="🔥",
        layout="centered",
    )

    # ---------- Custom CSS ----------
    st.markdown(
        """
        <style>
        body {
            background-color: #020617;
            color: #e5e7eb;
        }
        h1, h2, h3 {
            color: #fb923c;
        }
        .card {
            background: #020617;
            padding: 1.4rem;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
            margin-bottom: 1.6rem;
        }
        .stButton>button {
            background-color: #fb923c;
            color: black;
            border-radius: 12px;
            padding: 0.6rem 1.4rem;
            font-weight: 600;
        }
        .stDownloadButton>button {
            border-radius: 12px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Header ----------
    st.markdown(
        "<h1 style='text-align:center;'>🔥 Wildfire Detection System</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;color:#94a3b8;'>Deep learning–based wildfire detection using YOLO</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ---------- Logo ----------
    logos = glob("dalle-logos/*.png")
    if logos:
        st.image(random.choice(logos), use_column_width=True)

    # ---------- Model Configuration ----------
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🧠 Model Selection")

        col1, col2 = st.columns(2)
        with col1:
            model_type = st.radio(
                "Detection Type",
                ("Fire Detection", "General Object Detection"),
                horizontal=True,
            )

        models_dir = "general-models" if model_type == "General Object Detection" else "fire-models"
        model_files = sorted(
            f.replace(".pt", "")
            for f in os.listdir(models_dir)
            if f.endswith(".pt")
        )

        with col2:
            selected_model = st.selectbox("YOLO Model Variant", model_files)

        model_path = os.path.join(models_dir, selected_model + ".pt")
        model = load_model(model_path)

        st.caption(f"Loaded model: `{selected_model}.pt`")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Detection Parameters ----------
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🎛 Detection Parameters")

        col1, col2 = st.columns(2)
        with col1:
            conf_threshold = st.slider(
                "Confidence Threshold",
                0.0, 1.0, 0.25, 0.05,
                help="Minimum confidence required to display a detection."
            )
        with col2:
            iou_threshold = st.slider(
                "IOU Threshold",
                0.0, 1.0, 0.5, 0.05,
                help="Controls how overlapping boxes are filtered."
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Image Input ----------
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🖼 Input Image")

        image = None
        source = st.radio(
            "Choose Image Source",
            ("Upload from Computer", "Enter Image URL"),
            horizontal=True,
        )

        if source == "Upload from Computer":
            file = st.file_uploader(
                "Upload an image (JPG / PNG)",
                type=["jpg", "jpeg", "png"],
            )
            if file:
                image = Image.open(file)
        else:
            url = st.text_input("Paste direct image URL")
            if url:
                try:
                    r = requests.get(url, stream=True, timeout=10)
                    if r.status_code == 200:
                        image = Image.open(r.raw)
                    else:
                        st.error("Unable to load image from URL.")
                except Exception:
                    st.error("Invalid or unreachable image URL.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Prediction ----------
    if image is None:
        st.info("⬆ Upload an image or provide a valid image URL to start detection.")
    else:
        with st.spinner("🔥 Analyzing image for wildfire patterns..."):
            prediction, summary = predict_image(
                model, image, conf_threshold, iou_threshold
            )

        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📊 Detection Result")

            st.image(prediction, use_column_width=True)
            st.success(summary)

            buffer = io.BytesIO()
            Image.fromarray(prediction).save(buffer, format="PNG")

            st.download_button(
                "⬇ Download Result Image",
                buffer.getvalue(),
                file_name="wildfire_prediction.png",
                mime="image/png",
            )

            st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Footer ----------
    st.markdown(
        "<p style='text-align:center;color:#64748b;margin-top:2.5rem;'>Academic Project • AI & Computer Vision • YOLO-based Detection</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()


