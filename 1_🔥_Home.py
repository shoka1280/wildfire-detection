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


# -------------------------
# Model utilities
# -------------------------
def load_model(model_path):
    return YOLO(model_path)


def predict_image(model, image, conf_threshold, iou_threshold):
    res = model.predict(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        device="cpu",
    )

    class_name = model.model.names
    classes = res[0].boxes.cls
    class_counts = {}

    for c in classes:
        c = int(c)
        class_counts[class_name[c]] = class_counts.get(class_name[c], 0) + 1

    if not class_counts:
        text = "No wildfire-related objects detected."
    else:
        text = "Detected "
        for k, v in class_counts.items():
            text += f"{v} {k}{'s' if v > 1 else ''}, "
        text = text[:-2] + "."

    img = res[0].plot()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img, text


# -------------------------
# Main App
# -------------------------
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
        .main {
            background-color: #0f172a;
            color: #e5e7eb;
        }
        h1, h2, h3 {
            color: #f97316;
        }
        .stButton>button {
            background-color: #f97316;
            color: white;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
        }
        .stDownloadButton>button {
            border-radius: 10px;
        }
        .card {
            background: #020617;
            padding: 1.2rem;
            border-radius: 16px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.4);
            margin-bottom: 1.5rem;
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
        "<p style='text-align:center;color:#94a3b8;'>AI-based wildfire detection using YOLO</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ---------- Logo ----------
    logos = glob("dalle-logos/*.png")
    if logos:
        st.image(random.choice(logos), use_column_width=True)

    # ---------- Model Section ----------
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🧠 Model Configuration")

        col1, col2 = st.columns(2)
        with col1:
            model_type = st.radio(
                "Model Type",
                ("Fire Detection", "General"),
                horizontal=True,
            )

        models_dir = "general-models" if model_type == "General" else "fire-models"
        model_files = sorted(
            [f.replace(".pt", "") for f in os.listdir(models_dir) if f.endswith(".pt")]
        )

        with col2:
            selected_model = st.selectbox("Model Size", model_files)

        model = load_model(os.path.join(models_dir, selected_model + ".pt"))
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Thresholds ----------
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🎛 Detection Parameters")

        col1, col2 = st.columns(2)
        with col1:
            conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
        with col2:
            iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.5, 0.05)

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Image Input ----------
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🖼 Image Input")

        image = None
        source = st.radio(
            "Select Image Source",
            ("Upload from Computer", "Enter Image URL"),
            horizontal=True,
        )

        if source == "Upload from Computer":
            file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
            if file:
                image = Image.open(file)
        else:
            url = st.text_input("Direct Image URL")
            if url:
                try:
                    r = requests.get(url, stream=True, timeout=10)
                    if r.status_code == 200:
                        image = Image.open(r.raw)
                except Exception:
                    st.error("Invalid image URL.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Prediction ----------
    if image is None:
        st.info("⬆ Upload an image or paste a direct image URL to begin detection.")
    else:
        with st.spinner("🔥 Analyzing image for wildfire..."):
            result, text = predict_image(
                model, image, conf_threshold, iou_threshold
            )

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📊 Detection Result")
        st.image(result, use_column_width=True)
        st.success(text)

        buffer = io.BytesIO()
        Image.fromarray(result).save(buffer, format="PNG")

        st.download_button(
            "⬇ Download Result",
            buffer.getvalue(),
            file_name="wildfire_prediction.png",
            mime="image/png",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Footer ----------
    st.markdown(
        "<p style='text-align:center;color:#64748b;margin-top:2rem;'>Built for academic demonstration • AI & Computer Vision</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

