import streamlit as st
import numpy as np
import librosa
import joblib
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import io
import os
import tempfile

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CardioSense AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  –  deep navy / crimson / gold
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #050d1a;
    color: #e8eaf0;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: #080f1f !important;
    border-right: 1px solid #1a2540;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0a1628 0%, #0f2044 50%, #0a1628 100%);
    border: 1px solid #1e3060;
    border-radius: 20px;
    padding: 3rem 2.5rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(200,30,60,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 40px;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(212,175,55,0.1) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(90deg, #ffffff 0%, #d4af37 60%, #c81e3c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem 0;
    line-height: 1.1;
}
.hero-sub {
    color: #8899bb;
    font-size: 1.05rem;
    font-weight: 300;
    letter-spacing: 0.04em;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(200,30,60,0.15);
    border: 1px solid rgba(200,30,60,0.4);
    color: #ff6b85;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    margin-bottom: 1rem;
}

/* ── Cards ── */
.card {
    background: #0a1628;
    border: 1px solid #1a2a48;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: #d4af37;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Result box ── */
.result-box {
    border-radius: 18px;
    padding: 2rem 2.5rem;
    margin: 1.5rem 0;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-normal {
    background: linear-gradient(135deg, #051a0d 0%, #0a2e18 100%);
    border: 2px solid #1a7a40;
}
.result-abnormal {
    background: linear-gradient(135deg, #1a050a 0%, #2e0a14 100%);
    border: 2px solid #c81e3c;
}
.result-label {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 900;
    text-transform: capitalize;
    margin: 0;
}
.result-confidence {
    font-size: 1rem;
    color: #8899bb;
    margin-top: 0.4rem;
    font-weight: 300;
}

/* ── Probability bars ── */
.prob-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.65rem;
    gap: 0.75rem;
}
.prob-label {
    width: 90px;
    font-size: 0.82rem;
    font-weight: 500;
    color: #99aacc;
    text-align: right;
    text-transform: capitalize;
}
.prob-bar-wrap {
    flex: 1;
    background: #0f1e35;
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.6s ease;
}
.prob-pct {
    width: 44px;
    font-size: 0.82rem;
    color: #d4af37;
    font-weight: 600;
    text-align: right;
}

/* ── Info pills (sidebar) ── */
.info-pill {
    background: #0f1e35;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
    color: #8899bb;
}
.info-pill strong { color: #d4af37; }

/* ── Class descriptions ── */
.class-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #0f1e35;
    border: 1px solid #1a2a48;
    border-radius: 8px;
    padding: 0.45rem 0.8rem;
    font-size: 0.8rem;
    color: #aabbd4;
    margin: 0.25rem;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: #080f1f !important;
    border: 2px dashed #1a3060 !important;
    border-radius: 14px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #c81e3c 0%, #8b0e24 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #e0254a 0%, #a01028 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(200,30,60,0.35) !important;
}

/* ── Metric tiles ── */
.metric-tile {
    background: #0a1628;
    border: 1px solid #1a2a48;
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #d4af37;
}
.metric-key {
    font-size: 0.75rem;
    color: #5566880;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #5a6e90;
    margin-top: 0.2rem;
}

/* ── Divider ── */
hr { border-color: #1a2540 !important; }

/* ── Waveform plot background ── */
.stPlotlyChart, .stpyplot { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODELS  (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)
    model   = joblib.load(os.path.join(base, "heart_sound_rf_model.pkl"))
    scaler  = joblib.load(os.path.join(base, "scaler.pkl"))
    encoder = joblib.load(os.path.join(base, "label_encoder.pkl"))
    return model, scaler, encoder

model, scaler, label_encoder = load_models()
CLASS_NAMES = list(label_encoder.classes_)          # ['artifact','extrahls','extrastole','murmur','normal']


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_features(audio_path: str, sr: int = 22050, n_mfcc: int = 13) -> np.ndarray:
    """
    Extract 24 features from a heart-sound audio file.
    Matches the training pipeline: 13 MFCCs (mean) + 13 MFCCs (std)
    then trimmed / padded to 24 dimensions.
    """
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    # Normalise amplitude
    if y.max() != 0:
        y = y / np.max(np.abs(y))

    mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean   = np.mean(mfcc, axis=1)     # 13 values
    mfcc_std    = np.std(mfcc, axis=1)      # 13 values

    chroma      = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma)           # 1 value

    zcr         = np.mean(librosa.feature.zero_crossing_rate(y))   # 1 value
    rms         = np.mean(librosa.feature.rms(y=y))                # 1 value

    raw = np.concatenate([mfcc_mean, mfcc_std, [chroma_mean], [zcr], [rms]])

    # Pad / trim to exactly 24 features
    if len(raw) < 24:
        raw = np.pad(raw, (0, 24 - len(raw)))
    else:
        raw = raw[:24]

    return raw, y, sr


def predict(features: np.ndarray):
    scaled = scaler.transform(features.reshape(1, -1))
    proba  = model.predict_proba(scaled)[0]
    idx    = np.argmax(proba)
    label  = label_encoder.inverse_transform([idx])[0]
    return label, proba


# ─────────────────────────────────────────────
# WAVEFORM PLOT
# ─────────────────────────────────────────────
def plot_waveform(y: np.ndarray, sr: int):
    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor('#050d1a')
    ax.set_facecolor('#080f1f')

    times = np.linspace(0, len(y) / sr, len(y))
    ax.plot(times, y, color='#c81e3c', linewidth=0.7, alpha=0.9)
    ax.fill_between(times, y, 0, color='#c81e3c', alpha=0.08)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#1a2a48')
    ax.spines['left'].set_color('#1a2a48')
    ax.tick_params(colors='#5a6e90', labelsize=8)
    ax.set_xlabel("Time (s)", color='#5a6e90', fontsize=9)
    ax.set_ylabel("Amplitude", color='#5a6e90', fontsize=9)
    ax.set_title("Waveform", color='#8899bb', fontsize=10, pad=8)
    plt.tight_layout()
    return fig


def plot_spectrogram(y: np.ndarray, sr: int):
    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor('#050d1a')
    ax.set_facecolor('#080f1f')

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz',
                                   ax=ax, cmap='inferno')
    fig.colorbar(img, ax=ax, format='%+2.0f dB',
                 label='dB').ax.yaxis.label.set_color('#5a6e90')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#1a2a48')
    ax.spines['left'].set_color('#1a2a48')
    ax.tick_params(colors='#5a6e90', labelsize=8)
    ax.set_xlabel("Time (s)", color='#5a6e90', fontsize=9)
    ax.set_ylabel("Hz", color='#5a6e90', fontsize=9)
    ax.set_title("Spectrogram", color='#8899bb', fontsize=10, pad=8)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.5rem 0 1.5rem 0;'>
        <p style='font-family:Playfair Display,serif;font-size:1.4rem;font-weight:700;
                  color:#d4af37;margin:0;'>🫀 CardioSense</p>
        <p style='color:#3a4e6e;font-size:0.75rem;margin:0;letter-spacing:0.1em;'>AI HEART DIAGNOSTICS</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Model Overview**")
    st.markdown("""
    <div class='info-pill'>🤖 <strong>Algorithm:</strong> Random Forest</div>
    <div class='info-pill'>🌲 <strong>Estimators:</strong> 100 trees</div>
    <div class='info-pill'>📐 <strong>Features:</strong> 24 (MFCC + audio)</div>
    <div class='info-pill'>🎯 <strong>Classes:</strong> 5 categories</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Detectable Conditions**")

    conditions = {
        "🟢 Normal":      "Healthy heart sounds — regular S1/S2 pattern.",
        "🔴 Murmur":      "Turbulent blood flow; may indicate valve issues.",
        "🟡 Extrastole":  "Extra heartbeat (premature contraction).",
        "🟠 Extrahls":    "Extra heart sounds (S3/S4 gallop).",
        "⚫ Artifact":    "Recording noise or non-cardiac sound.",
    }
    for name, desc in conditions.items():
        with st.expander(name):
            st.caption(desc)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem;color:#3a4e6e;text-align:center;padding-top:0.5rem;'>
    ⚕️ For clinical reference only.<br>Always consult a cardiologist.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-badge'>AI-Powered Cardiac Analysis</div>
    <h1 class='hero-title'>CardioSense AI</h1>
    <p class='hero-sub'>Upload a heart sound recording and receive instant classification<br>
    powered by a trained Random Forest model.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1.3], gap="large")

with col_upload:
    st.markdown("<div class='card-title'>📁 Upload Audio</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        label="Drag & drop or click to browse",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        help="Supported: WAV, MP3, FLAC, OGG, M4A",
    )

    if uploaded:
        st.audio(uploaded, format="audio/wav")

        # Audio metadata (quick peek)
        with tempfile.NamedTemporaryFile(suffix=f".{uploaded.name.split('.')[-1]}", delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        y_raw, sr_raw = librosa.load(tmp_path, sr=None)
        duration = len(y_raw) / sr_raw

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class='metric-tile'>
                <div class='metric-val'>{duration:.1f}s</div>
                <div class='metric-key'>Duration</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class='metric-tile'>
                <div class='metric-val'>{sr_raw//1000}k</div>
                <div class='metric-key'>Sample Rate</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class='metric-tile'>
                <div class='metric-val'>{len(y_raw)//1000}K</div>
                <div class='metric-key'>Samples</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        analyse_btn = st.button("🫀  Analyse Heart Sound")
    else:
        analyse_btn = False
        tmp_path    = None

        # Placeholder instructions
        st.markdown("""
        <div class='card' style='text-align:center;padding:2rem;'>
            <div style='font-size:3rem;margin-bottom:1rem;'>🎙️</div>
            <p style='color:#5a6e90;font-size:0.9rem;margin:0;'>
                Upload a <strong style='color:#8899bb;'>WAV / MP3 / FLAC</strong> heart sound recording<br>
                to begin AI-powered classification.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# RESULTS PANEL
# ─────────────────────────────────────────────
with col_result:
    st.markdown("<div class='card-title'>📊 Analysis Results</div>", unsafe_allow_html=True)

    if uploaded and analyse_btn:
        with st.spinner("Extracting features and running inference…"):
            try:
                features, y_sig, sr_sig = extract_features(tmp_path)
                label, probabilities    = predict(features)
            except Exception as e:
                st.error(f"Error processing audio: {e}")
                st.stop()

        is_normal   = label.lower() == "normal"
        box_class   = "result-normal" if is_normal else "result-abnormal"
        label_color = "#2ecc71" if is_normal else "#ff4466"
        confidence  = np.max(probabilities) * 100
        icon        = "✅" if is_normal else "⚠️"

        st.markdown(f"""
        <div class='result-box {box_class}'>
            <div style='font-size:2.5rem;margin-bottom:0.3rem;'>{icon}</div>
            <p class='result-label' style='color:{label_color};'>{label.upper()}</p>
            <p class='result-confidence'>Confidence: <strong style='color:#d4af37;'>{confidence:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # ── Per-class probability bars ──
        st.markdown("<div class='card-title' style='margin-top:1rem;'>Class Probabilities</div>",
                    unsafe_allow_html=True)

        bar_colors = {
            "normal":     "#2ecc71",
            "murmur":     "#e74c3c",
            "extrastole": "#f39c12",
            "extrahls":   "#3498db",
            "artifact":   "#9b59b6",
        }

        for cls, prob in zip(CLASS_NAMES, probabilities):
            pct   = prob * 100
            color = bar_colors.get(cls, "#d4af37")
            st.markdown(f"""
            <div class='prob-row'>
                <span class='prob-label'>{cls}</span>
                <div class='prob-bar-wrap'>
                    <div class='prob-bar-fill'
                         style='width:{pct:.1f}%;background:{color};'></div>
                </div>
                <span class='prob-pct'>{pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class='card' style='text-align:center;padding:3rem 2rem;'>
            <div style='font-size:2.5rem;margin-bottom:1rem;'>📈</div>
            <p style='color:#5a6e90;font-size:0.9rem;'>
                Classification results and probability<br>
                scores will appear here after analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# VISUALISATIONS (shown after analysis)
# ─────────────────────────────────────────────
if uploaded and analyse_btn and 'y_sig' in dir():
    st.markdown("---")
    st.markdown("<div class='card-title' style='font-size:1.2rem;'>🔬 Signal Visualisation</div>",
                unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🌊 Waveform", "🎨 Spectrogram"])
    with tab1:
        fig_w = plot_waveform(y_sig, sr_sig)
        st.pyplot(fig_w, use_container_width=True)
        plt.close(fig_w)
    with tab2:
        fig_s = plot_spectrogram(y_sig, sr_sig)
        st.pyplot(fig_s, use_container_width=True)
        plt.close(fig_s)

    # ── MFCC heatmap ──
    st.markdown("<div class='card-title' style='margin-top:1rem;'>🎛️ MFCC Feature Map</div>",
                unsafe_allow_html=True)
    fig_m, ax_m = plt.subplots(figsize=(10, 2.8))
    fig_m.patch.set_facecolor('#050d1a')
    ax_m.set_facecolor('#080f1f')
    mfcc_map = librosa.feature.mfcc(y=y_sig, sr=sr_sig, n_mfcc=13)
    img2 = librosa.display.specshow(mfcc_map, x_axis='time', ax=ax_m, cmap='magma')
    fig_m.colorbar(img2, ax=ax_m)
    ax_m.set_title("MFCC Coefficients (1–13)", color='#8899bb', fontsize=10)
    ax_m.tick_params(colors='#5a6e90', labelsize=8)
    for sp in ax_m.spines.values():
        sp.set_color('#1a2a48')
    plt.tight_layout()
    st.pyplot(fig_m, use_container_width=True)
    plt.close(fig_m)


# ─────────────────────────────────────────────
# DISCLAIMER FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;padding:0.5rem 0 1rem 0;'>
    <p style='color:#2a3a58;font-size:0.75rem;'>
        ⚕️ CardioSense AI is a research tool and does <strong>not</strong> replace professional medical advice.
        Always consult a qualified cardiologist for clinical decisions.
    </p>
    <p style='color:#1a2a40;font-size:0.7rem;'>Built with Streamlit · Random Forest · Librosa · scikit-learn</p>
</div>
""", unsafe_allow_html=True)
