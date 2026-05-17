import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import joblib
from datetime import datetime
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Heart Sound AI",
    page_icon="🫀",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background: linear-gradient(to bottom right, #020617, #0f172a);
    color: white;
}

.main-title {
    font-size: 54px;
    font-weight: 900;
    text-align: center;
    color: #7CFFB2;
    margin-top: 10px;
}

.sub-title {
    text-align: center;
    color: #cbd5e1;
    font-size: 20px;
    margin-bottom: 40px;
}

.result-box {
    background: linear-gradient(135deg,#111827,#1e293b);
    border-radius: 30px;
    padding: 40px;
    text-align: center;
    border: 1px solid rgba(0,255,170,0.25);
    box-shadow: 0 0 35px rgba(0,255,170,0.15);
    margin-bottom: 30px;
}

.prediction-text {
    font-size: 65px;
    font-weight: 900;
    color: white;
    text-shadow: 0 0 25px rgba(0,255,170,0.7);
}

.confidence-text {
    color: #7CFFB2;
    font-size: 24px;
    font-weight: 600;
}

.card {
    background: rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 20px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.08);
}

.stButton>button {
    width: 100%;
    height: 55px;
    border-radius: 15px;
    border: none;
    background: linear-gradient(90deg,#00ffae,#00c3ff);
    color: black;
    font-weight: bold;
    font-size: 18px;
}

.stDownloadButton>button {
    width: 100%;
    height: 50px;
    border-radius: 15px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD FILES
# =====================================================

model = joblib.load("heart_sound_rf_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# =====================================================
# PREPROCESSING
# =====================================================

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

# =====================================================
# FEATURE EXTRACTION
# =====================================================

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

# =====================================================
# PDF REPORT
# =====================================================

def generate_pdf(prediction, results):

    pdf_path = "Heart_Report.pdf"

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

# =====================================================
# HEADER
# =====================================================

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

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("## 📂 Upload Heart Sound")

    uploaded_file = st.file_uploader(
        "Upload WAV file",
        type=["wav"]
    )

    st.markdown("---")

    st.markdown("""
    ### 🩺 About
    
    AI-powered phonocardiogram (PCG)
    analysis system for detecting
    abnormal heart sounds using
    machine learning.
    """)

# =====================================================
# MAIN LOGIC
# =====================================================

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

    # =====================================================
    # RESULT CARD
    # =====================================================

    st.markdown(f"""
    <div class="result-box">

        <h1 style="color:#7CFFB2;">
        🫀 Predicted Heart Condition
        </h1>

        <div class="prediction-text">
        {predicted_label.upper()}
        </div>

        <br>

        <div class="confidence-text">
        Confidence Score: {confidence:.2f}%
        </div>

    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # CHARTS
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("## 📈 Waveform")

        fig, ax = plt.subplots(figsize=(10,3))

        librosa.display.waveshow(
            signal,
            sr=sr,
            ax=ax
        )

        ax.set_title("Preprocessed Heart Sound")

        st.pyplot(fig)

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

    # =====================================================
    # PROBABILITIES
    # =====================================================

    st.markdown("## 🏆 Prediction Probabilities")

    for cls, prob in results:

        st.markdown(f"""
        <div class="card">
        <h3>{cls.upper()}</h3>
        <h4>{prob*100:.2f}% probability</h4>
        </div>
        """, unsafe_allow_html=True)

        st.progress(float(prob))

    # =====================================================
    # HEALTH ADVISORY
    # =====================================================

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

    # =====================================================
    # AUDIO PLAYER
    # =====================================================

    st.markdown("## 🔊 Uploaded Audio")

    st.audio(uploaded_file)

    # =====================================================
    # PDF DOWNLOAD
    # =====================================================

    pdf_path = generate_pdf(
        predicted_label,
        results
    )

    with open(pdf_path, "rb") as f:

        st.download_button(
            label="📥 Download Medical Report",
            data=f,
            file_name="Heart_Report.pdf",
            mime="application/pdf"
        )
