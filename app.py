import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ── Page config ──
st.set_page_config(
    page_title="PRE-SLIP Calculator",
    page_icon="🦴",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Hide sidebar, header, footer ──
st.markdown("""
<style>
    [data-testid="collapsedControl"] { display: none; }
    [data-testid="stSidebar"] { display: none; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
    footer { visibility: hidden; }

    /* Custom header */
    .main-header {
        text-align: center;
        padding: 2rem 1rem 1rem;
    }
    .main-header h1 {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1B5E7B;
        margin-bottom: 0.2rem;
        letter-spacing: 0.02em;
    }
    .main-header p {
        font-size: 0.88rem;
        color: #666;
        margin-top: 0;
    }
    .badge-row {
        display: flex;
        gap: 8px;
        justify-content: center;
        margin-top: 12px;
        flex-wrap: wrap;
    }
    .badge {
        background: #E8F2F7;
        color: #1B5E7B;
        border-radius: 16px;
        padding: 4px 14px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }

    /* Card styling */
    .input-section {
        background: #FAFAF8;
        border: 1px solid #E8E6E0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .section-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #1B5E7B;
        margin-bottom: 1rem;
        padding-left: 8px;
        border-left: 3px solid #1B5E7B;
    }

    /* Result phenotype card */
    .pheno-card {
        text-align: center;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    .pheno-card .label {
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #888;
    }
    .pheno-card .name {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .pheno-card .conf {
        font-size: 0.9rem;
        color: #666;
        font-family: 'Courier New', monospace;
    }
    .compensated-card { background: rgba(0,114,178,0.06); border: 1.5px solid rgba(0,114,178,0.2); }
    .compensated-card .name { color: #0072B2; }
    .rigid-card { background: rgba(213,94,0,0.06); border: 1.5px solid rgba(213,94,0,0.2); }
    .rigid-card .name { color: #D55E00; }
    .unstable-card { background: rgba(0,158,115,0.06); border: 1.5px solid rgba(0,158,115,0.2); }
    .unstable-card .name { color: #009E73; }

    /* Disclaimer */
    .disclaimer {
        text-align: center;
        font-size: 0.7rem;
        color: #999;
        margin-top: 2rem;
        padding-top: 1.5rem;
        border-top: 1px solid #E8E6E0;
        line-height: 1.6;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #1B5E7B, #1A7A5A) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.65rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        letter-spacing: 0.03em !important;
        transition: transform 0.15s, box-shadow 0.2s !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(27,94,123,0.3) !important;
    }

    /* Number input styling */
    [data-testid="stNumberInput"] input {
        border-radius: 8px !important;
        font-family: 'Courier New', monospace !important;
    }
    [data-testid="stSelectbox"] > div > div {
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Data ──
DATA_DIR = Path(__file__).parent / "streamlit_data"
FEATURE_NAMES = ["Age", "BMI", "Gender", "VAS", "LL", "SS"]
CLASS_NAMES = ["Compensated", "Rigid", "Unstable"]
COLORS = {"Compensated": "#0072B2", "Rigid": "#D55E00", "Unstable": "#009E73"}


@st.cache_resource
def load_model():
    pipeline = joblib.load(DATA_DIR / "lr_pipeline.joblib")
    scaler = joblib.load(DATA_DIR / "scaler.joblib")
    lr_model = joblib.load(DATA_DIR / "lr_model.joblib")
    X_df = pd.read_csv(DATA_DIR / "all_features.csv")
    return pipeline, scaler, lr_model, X_df


pipeline, scaler, lr_model, X_df = load_model()

# ── Header ──
st.markdown("""
<div class="main-header">
    <h1>PRE-SLIP Calculator</h1>
    <p>Preoperative Phenotype Screening for Lumbar Isthmic Spondylolisthesis</p>
    <div class="badge-row">
        <span class="badge">Logistic Regression</span>
        <span class="badge">10-Fold CV · AUC 0.881</span>
        <span class="badge">n = 501</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Input section ──
st.markdown('<div class="section-label">Patient Parameters</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age (years)", value=62, min_value=18, max_value=100, step=1,
                          help="Cohort range: 44 – 80")
with col2:
    gender = st.selectbox("Gender", options=["Female", "Male"],
                          help="0 = Female, 1 = Male")
with col3:
    bmi = st.number_input("BMI (kg/m²)", value=25.4, min_value=10.0, max_value=60.0,
                          step=0.1, format="%.1f", help="Cohort range: 17.6 – 38.7")

col4, col5, col6 = st.columns(3)
with col4:
    vas = st.selectbox("VAS (0–10)", options=list(range(11)), index=3,
                       help="Visual Analogue Scale for pain")
with col5:
    ll = st.number_input("Lumbar Lordosis LL (°)", value=47.2, min_value=0.0,
                         max_value=120.0, step=0.1, format="%.1f",
                         help="Cohort range: 6.7 – 105.5")
with col6:
    ss = st.number_input("Sacral Slope SS (°)", value=37.8, min_value=0.0,
                         max_value=80.0, step=0.1, format="%.1f",
                         help="Cohort range: 8.9 – 66.8")

st.write("")
predict_clicked = st.button("Calculate Phenotype", use_container_width=True)

# ── Prediction ──
if predict_clicked:
    gender_val = 1 if gender == "Male" else 0
    patient = np.array([[age, bmi, gender_val, vas, ll, ss]])

    pred_idx = pipeline.predict(patient)[0]
    pred_proba = pipeline.predict_proba(patient)[0]
    pred_label = CLASS_NAMES[pred_idx]
    pred_conf = pred_proba[pred_idx]

    st.markdown("---")

    # Phenotype result card
    card_class = pred_label.lower() + "-card"
    st.markdown(f"""
    <div class="pheno-card {card_class}">
        <div class="label">Predicted phenotype</div>
        <div class="name">{pred_label}</div>
        <div class="conf">Probability {pred_conf:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

    # Probability bars
    st.markdown('<div class="section-label">Phenotype Probabilities</div>',
                unsafe_allow_html=True)

    for cls, prob in zip(CLASS_NAMES, pred_proba):
        c1, c2, c3 = st.columns([2, 6, 1.5])
        with c1:
            st.markdown(f"**{cls}**")
        with c2:
            st.progress(float(prob))
        with c3:
            st.markdown(f"`{prob:.1%}`")

    # SHAP waterfall
    st.markdown("---")
    st.markdown('<div class="section-label">SHAP Explanation</div>',
                unsafe_allow_html=True)
    st.caption(
        "Waterfall plots decompose the prediction for each phenotype. "
        "Red = pushes probability higher; Blue = pushes probability lower."
    )

    patient_scaled = scaler.transform(patient)
    X_background = scaler.transform(X_df.values)
    explainer = shap.LinearExplainer(lr_model, X_background,
                                     feature_names=FEATURE_NAMES)
    sv = explainer.shap_values(patient_scaled)

    shap_cols = st.columns(3)
    for i, cls in enumerate(CLASS_NAMES):
        with shap_cols[i]:
            st.markdown(f"**{cls}**")
            explanation = shap.Explanation(
                values=sv[0, :, i],
                base_values=explainer.expected_value[i],
                data=patient[0],
                feature_names=FEATURE_NAMES,
            )
            fig, ax = plt.subplots(figsize=(4, 3))
            shap.plots.waterfall(explanation, max_display=6, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ── Disclaimer ──
st.markdown("""
<div class="disclaimer">
    <strong>Disclaimer</strong> — For research and educational purposes only.
    Not a substitute for professional clinical judgment.<br>
    External validation is recommended before clinical adoption.<br><br>
</div>
""", unsafe_allow_html=True)
