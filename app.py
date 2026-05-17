import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import joblib
import pandas as pd
from datetime import datetime
from matplotlib.patches import Wedge
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Heart Sound AI",
    page_icon="🫀",
    layout="wide"
)

# =========================================
# CUSTOM CSS
# =========================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to bottom right, #050816, #0b1120);
    color: white;
}

.main-title {
    font-size: 48px;
    font-weight: 800;
    color: #7CFFB2;
    text-align: center;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 30px;
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 0 20px rgba(0,255,170,0.08);
    margin-bottom: 20px;
}

.result-card {
    background: linear-gradient(135deg, #0f172a, #111827);
    border-radius: 25px;
    padding: 30px;
    text-align: center;
    border: 1px solid rgba(0,255,170,0.25);
    box-shadow: 0 0 25px rgba(0,255,170,0.15);
}

.result-label {
    color: #7CFFB2;
    font-size: 22px;
    font-weight: 600;
}

.result-prediction {
    font-size: 48px;
    font-weight: 900;
    color: white;
}

.metric-card {
    background: rgba(255,255,255,0.04);
    padding: 18px;
    border-radius: 18px;
    margin-bottom: 15px;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(90deg,#00ffae,#00c3ff);
    color: black;
    border-radius: 15px;
    font-weight: bold;
    height: 55px;
    border: none;
    font-size: 18px;
}

.stDownloadButton>button {
    width: 100%;
    border-radius: 15px;
    height: 50px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD MODEL FILES
# =========================================

model = joblib.load("heart_sound_rf_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# =========================================
# PREPROCESSING
# =========================================

def preprocess_signal(file_path, duration=5):

    signal, sr = librosa.load(
        file_path,
        sr=22050,
        duration=duration
    )

    max_val = np.max(np.abs(signal))

    if max_val != 0:
        signal = signal / max_val

    signal, _ = librosa.effects.trim(signal)

    return signal, sr

# =========================================
# FEATURE EXTRACTION
# =========================================

def extract_features(signal, sr):

    features = []

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=20,
        n_fft=2048,
        hop_length=512
    )

    mfcc_mean = np.mean(mfcc, axis=1)

    features.extend(mfcc_mean)

    rms = librosa.feature.rms(y=signal)
    features.append(np.mean(rms))

    centroid = librosa.feature.spectral_centroid(
        y=signal,
        sr=sr
    )
    features.append(np.mean(centroid))

    bandwidth = librosa.feature.spectral_bandwidth(
        y=signal,
        sr=sr
    )
    features.append(np.mean(bandwidth))

    zcr = librosa.feature.zero_crossing_rate(signal)
    features.append(np.mean(zcr))

    return features

# =========================================
# PDF REPORT GENERATION
# =========================================

def generate_pdf(prediction, results):

    pdf_path = "HeartSound_Report.pdf"

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter
    )

    styles = getSampleStyleSheet()

    elements = []

    title = Paragraph(
        "<b>Heart Sound Classification Report</b>",
        styles['Title']
    )

    elements.append(title)
    elements.append(Spacer(1,20))

    date = Paragraph(
        f"Generated on: {datetime.now()}",
        styles['Normal']
    )

    elements.append(date)
    elements.append(Spacer(1,20))

    pred = Paragraph(
        f"<b>Predicted Condition:</b> {prediction}",
        styles['Heading2']
    )

    elements.append(pred)
    elements.append(Spacer(1,20))

    table_data = [["Condition","Probability"]]

    for cls, prob in results:
        table_data.append([
            cls,
            f"{prob*100:.2f}%"
        ])

    table = Table(table_data)

    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.darkblue),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('GRID',(0,0),(-1,-1),1,colors.black),
        ('BACKGROUND',(0,1),(-1,-1),colors.beige)
    ]))

    elements.append(table)

    doc.build(elements)

    return pdf_path

# =========================================
# HEADER
# =========================================

st.markdown("""
<div class="main-title">
🫀 Heart Sound Classification AI
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sub-title">
AI-powered PCG Signal Analysis & Abnormality Detection System
</div>
""", unsafe_allow_html=True)

# =========================================
# SIDEBAR
# =========================================

with st.sidebar:

    st.markdown("## 📂 Upload Heart Sound")

    uploaded_file = st.file_uploader(
        "Upload WAV file",
        type=["wav"]
    )

# =========================================
# MAIN LOGIC
# =========================================

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:

        tmp_file.write(uploaded_file.read())

        temp_path = tmp_file.name

    signal, sr = preprocess_signal(temp_path)

    features = extract_features(signal, sr)

    features = np.array(features).reshape(1, -1)

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    predicted_label = le.inverse_transform(prediction)[0]

    probabilities = model.predict_proba(features_scaled)[0]

    class_names = le.classes_

    results = list(zip(class_names, probabilities))

    results = sorted(
        results,
        key=lambda x: x[1],
        reverse=True
    )

    confidence = results[0][1] * 100

    # =========================================
    # RESULT CARD
    # =========================================

   st.markdown(f"""
<div class="result-card">

    <div class="result-label">
        🫀 Predicted Heart Condition
    </div>

    <div style="
        font-size:60px;
        font-weight:900;
        color:#ffffff;
        margin-top:20px;
        margin-bottom:20px;
        text-shadow:0 0 25px rgba(0,255,170,0.6);
    ">
        {predicted_label.upper()}
    </div>

    <div style="
        font-size:24px;
        color:#7CFFB2;
        font-weight:600;
    ">
        Confidence Score: {confidence:.2f}%
    </div>

</div>
""", unsafe_allow_html=True)

    # =========================================
    # WAVEFORM
    # =========================================

    with col1:

        st.markdown("## 📈 Waveform")

        fig, ax = plt.subplots(figsize=(10,3))

        librosa.display.waveshow(
            signal,
            sr=sr,
            ax=ax
        )

        ax.set_title("Preprocessed PCG Signal")

        st.pyplot(fig)

    # =========================================
    # SPECTROGRAM
    # =========================================

    with col2:

        st.markdown("## 🌌 Spectrogram")

        spectrogram = librosa.stft(signal)

        spectrogram_db = librosa.amplitude_to_db(
            abs(spectrogram)
        )

        fig2, ax2 = plt.subplots(figsize=(10,3))

        img = librosa.display.specshow(
            spectrogram_db,
            sr=sr,
            x_axis='time',
            y_axis='hz',
            ax=ax2
        )

        plt.colorbar(img, ax=ax2)

        st.pyplot(fig2)

    # =========================================
    # TOP PREDICTIONS
    # =========================================

    st.markdown("## 🏆 Top Predictions")

    for cls, prob in results:

        st.markdown(f"""
        <div class="metric-card">
        <b>{cls.upper()}</b><br>
        {prob*100:.2f}% probability
        </div>
        """, unsafe_allow_html=True)

        st.progress(float(prob))

    # =========================================
    # ADVISORY
    # =========================================

    st.markdown("## 🩺 AI Health Advisory")

    if predicted_label == "normal":

        st.success("""
        Heart sound appears normal.
        No major abnormality detected.
        """)

    else:

        st.warning(f"""
        Detected pattern resembles:
        {predicted_label.upper()}
        
        Recommend clinical verification
        by healthcare professional.
        """)

    # =========================================
    # AUDIO PLAYER
    # =========================================

    st.markdown("## 🔊 Uploaded Audio")

    st.audio(uploaded_file)

    # =========================================
    # PDF REPORT
    # =========================================

    pdf_path = generate_pdf(
        predicted_label,
        results
    )

    with open(pdf_path, "rb") as f:

        st.download_button(
            label="📥 Download Medical Report",
            data=f,
            file_name="HeartSound_Report.pdf",
            mime="application/pdf"
        )
