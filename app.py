# app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import io
from datetime import datetime
import requests

# ML
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mn_preprocess

# PDF
from fpdf import FPDF

# ------------------ Page Config ------------------
st.set_page_config(page_title="LeafIQ – Plant Health Detector", page_icon="🌿", layout="wide")

# ------------------ BlossomPlant Theme CSS ------------------
st.markdown("""
<style>
.title-container {
    display: flex;
    align-items: center;
    gap: 15px;
}
.title-text {
    font-size:32px; 
    font-weight:700; 
    color:#006D5F;
    line-height:1.2;
}
.subtitle {
    color:#00A387;
    margin-top:-10px;
    margin-bottom:15px;
}
.card {
    background:#C7F0E9;
    padding:20px;
    border-radius:14px;
    border:1px solid #00A387;
    box-shadow:0 6px 18px rgba(0,0,0,0.06);
}
body {
    background-color:#F9FDFB;
    color:#4A4A4A;
}
.stButton>button {
    background-color:#00A387;
    color:white; 
    border-radius:10px; 
    padding:10px 16px;
}
.stButton>button:hover {
    background-color:#006D5F;
    transform:scale(1.02);
}
</style>
""", unsafe_allow_html=True)

# ------------------ Logo + Title ------------------
logo_path = "LeafIQ_logo.png"

if os.path.exists(logo_path):
    col_logo, col_title = st.columns([0.3, 2])
    with col_logo:
        st.image(logo_path, use_container_width=True)
    with col_title:
        st.markdown(
            '<div class="title-text" style="margin-top:25px;">'
            'LeafIQ<br><span style="font-size:24px; font-weight:500;">Plant Disease Detector</span>'
            '</div>',
            unsafe_allow_html=True
        )
else:
    st.markdown(
        '<div class="title-text" style="margin-top:25px;">🌿 LeafIQ<br>'
        '<span style="font-size:24px; font-weight:500;">Plant Disease Detector</span></div>',
        unsafe_allow_html=True
    )

st.markdown('<div class="subtitle">Smart Leaf Health Scoring using MobileNetV2 + Image Processing</div>',
            unsafe_allow_html=True)
st.markdown("---")

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("About LeafIQ")
    st.write(
        "LeafIQ is a lightweight prototype that uses a MobileNetV2 feature extractor "
        "combined with visual heuristics to estimate leaf health."
    )
    st.markdown("---")
    st.header("Credits")
    st.write("Built by: **Vanshika Maru**")
    st.write("Tech: Streamlit, TensorFlow, OpenCV, Pillow")
    st.markdown("---")
    st.write("Supported formats: JPEG, PNG, WEBP, JFIF")

# ------------------ Helper Functions ------------------
def load_image_from_file(path_or_buffer):
    return Image.open(path_or_buffer).convert("RGB")

def compute_brown_ratio(pil_img):
    img = np.array(pil_img)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = np.array([5, 40, 20])
    upper = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return np.count_nonzero(mask) / (img.size/3), mask

def detect_spots(pil_img):
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    inv = 255 - th
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = img.shape[0]*img.shape[1]
    area_sum = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 50)
    return area_sum / total_area, contours

def mobilenet_features(pil_img, model):
    img = pil_img.resize((224,224))
    arr = mn_preprocess(np.array(img).astype(np.float32))
    arr = np.expand_dims(arr, 0)
    feat = model.predict(arr, verbose=0)
    return np.mean(feat, axis=(1,2)).flatten()

def severity_label(score):
    if score < 0.03: return "Healthy", "Low"
    if score < 0.09: return "Mild Infection", "Medium"
    if score < 0.20: return "Moderate Infection", "High"
    return "Severe Infection", "Very High"

def create_pdf(image_bytes, file_name, pred, conf, details):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0,10,"LeafIQ – Plant Report", ln=True, align="C")

    pdf.set_font("Arial", size=12)
    pdf.ln(5)
    pdf.cell(0,8,f"File: {file_name}", ln=True)
    pdf.cell(0,8,f"Prediction: {pred}", ln=True)
    pdf.cell(0,8,f"Confidence: {conf:.1f}%", ln=True)
    pdf.cell(0,8,f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    pdf.ln(5)
    pdf.multi_cell(0, 6, "Details:")
    for k,v in details.items():
        pdf.multi_cell(0, 6, f"- {k}: {v}")

    tmp = "temp_leaf.png"
    with open(tmp, "wb") as f:
        f.write(image_bytes)
    pdf.image(tmp, x=55, w=90)
    os.remove(tmp)

    out = io.BytesIO()
    out.write(pdf.output(dest="S").encode("latin-1"))
    out.seek(0)
    return out

# ------------------ Load MobileNetV2 ------------------
@st.cache_resource
def load_model():
    return MobileNetV2(include_top=False, weights="imagenet", input_shape=(224,224,3))

model = load_model()

# ------------------ Samples ------------------
SAMPLES = {
    "None": None,
    "Healthy Leaf": "Sample_images/healthy_leaf.webp",
    "Mild Leaf Spot": "Sample_images/Mild_spot_leaf.jfif",
    "Severe Leaf Spot": "Sample_images/Serve_spot_leaf.jfif",
}

# ------------------ Main Layout ------------------
left, right = st.columns([1.1, 1])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Upload / Samples")

    selected = st.selectbox("Choose a sample:", list(SAMPLES.keys()))
    uploaded = st.file_uploader(
        "Upload leaf image",
        type=["jpg", "jpeg", "png", "jfif", "webp"]
    )

    image = None
    img_name = "uploaded"

    if selected != "None" and uploaded is None:
        if os.path.exists(SAMPLES[selected]):
            image = load_image_from_file(SAMPLES[selected])
            img_name = os.path.basename(SAMPLES[selected])
    elif uploaded:
        image = load_image_from_file(uploaded)
        img_name = uploaded.name

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Analysis & Result")

    if image is not None:
        st.image(image, caption="Input Leaf", use_container_width=True)

        # Analysis
        brown_ratio, mask = compute_brown_ratio(image)
        spot_ratio, contours = detect_spots(image)
        feat = mobilenet_features(image, model)
        feat_norm = np.linalg.norm(feat) / (np.sqrt(len(feat)) + 1e-9)

        combined = 0.7*(0.6*brown_ratio + 0.4*spot_ratio) + 0.3*(0.0009*feat_norm)

        label, sev = severity_label(combined)
        conf = min(99.0, max(30.0, combined * 400))

        st.markdown(f"### 🌱 Result: **{label}**")
        st.metric("Severity Level", sev, delta=f"Confidence: {conf:.1f}%")

        st.write("### Details")
        st.write(f"- Brown Ratio: `{brown_ratio:.4f}`")
        st.write(f"- Spot Ratio: `{spot_ratio:.4f}`")
        st.write(f"- Feature Norm: `{feat_norm:.3f}`")
        st.write(f"- Combined Score: `{combined:.4f}`")

        # PDF Button
        if st.button("📄 Generate Report (PDF)"):
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            pdf = create_pdf(
                buf.getvalue(), img_name, label, conf,
                {
                    "Brown Ratio": brown_ratio,
                    "Spot Ratio": spot_ratio,
                    "Feature Norm": feat_norm,
                    "Score": combined,
                    "Severity": sev
                }
            )
            st.download_button(
                "Download PDF",
                data=pdf,
                file_name=f"LeafIQ_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

    else:
        st.info("Upload an image or select a sample to begin.")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("LeafIQ Prototype • Powered by MobileNetV2 • Built by Vanshika Maru")
