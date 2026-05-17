import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import joblib

# Load saved files
model = joblib.load("heart_sound_rf_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# -----------------------------------
# Preprocessing Function
# -----------------------------------

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

# -----------------------------------
# Feature Extraction Function
# -----------------------------------

def extract_features(signal, sr):

    features = []

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=20,
        n_fft=2048,
        hop_length=512
    )

    mfcc_mean = np.mean(mfcc, axis=1)

    features.extend(mfcc_mean)

    # RMS Energy
    rms = librosa.feature.rms(y=signal)
    features.append(np.mean(rms))

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(
        y=signal,
        sr=sr
    )

    features.append(np.mean(centroid))

    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(
        y=signal,
        sr=sr
    )

    features.append(np.mean(bandwidth))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(signal)

    features.append(np.mean(zcr))

    return features

# -----------------------------------
# Streamlit UI
# -----------------------------------

st.title("Heart Sound Classification using Machine Learning")

st.write("Upload a heart sound (.wav) file")

uploaded_file = st.file_uploader(
    "Choose a WAV file",
    type=["wav"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    # Save temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:

        tmp_file.write(uploaded_file.read())

        temp_path = tmp_file.name

    # Preprocess
    signal, sr = preprocess_signal(temp_path)

    # Show waveform
    fig, ax = plt.subplots(figsize=(10,3))

    librosa.display.waveshow(
        signal,
        sr=sr,
        ax=ax
    )

    ax.set_title("Preprocessed Heart Sound")

    st.pyplot(fig)

    # Feature extraction
    features = extract_features(signal, sr)

    features = np.array(features).reshape(1, -1)

    # Scaling
    features_scaled = scaler.transform(features)

    # Prediction
    prediction = model.predict(features_scaled)

    predicted_label = le.inverse_transform(prediction)

    st.subheader(f"Predicted Class: {predicted_label[0]}")

    # Probabilities
    probabilities = model.predict_proba(features_scaled)[0]

    class_names = le.classes_

    results = list(zip(class_names, probabilities))

    results = sorted(
        results,
        key=lambda x: x[1],
        reverse=True
    )

    st.subheader("Prediction Probabilities")

    for class_name, prob in results:

        st.write(f"{class_name} → {prob*100:.2f}%")