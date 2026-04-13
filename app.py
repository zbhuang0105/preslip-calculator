import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path

# ──────────────────────────────────────────────────────────
# Data initialisation (called once, cached in session_state)
# ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "streamlit_data"

FEATURE_NAMES = ["Age", "BMI", "Gender", "VAS", "LL", "SS"]
CLASS_NAMES   = ["Compensated", "Rigid", "Unstable"]

FEATURE_DISPLAY = {
    "Age":    "Age (years)",
    "BMI":    "Body Mass Index (kg/m²)",
    "Gender": "Gender (0 = Female, 1 = Male)",
    "VAS":    "Visual Analogue Scale (0–10)",
    "LL":     "Lumbar Lordosis (°)",
    "SS":     "Sacral Slope (°)",
}


def initialize_data():
    """Load model, scaler and reference data into session_state."""
    if "model" in st.session_state:
        return  # already loaded

    pipeline = joblib.load(DATA_DIR / "lr_pipeline.joblib")
    scaler   = joblib.load(DATA_DIR / "scaler.joblib")
    lr_model = joblib.load(DATA_DIR / "lr_model.joblib")

    X_df = pd.read_csv(DATA_DIR / "all_features.csv")
    labels_df = pd.read_csv(DATA_DIR / "all_labels.csv")

    with open(DATA_DIR / "shap_data.pkl", "rb") as f:
        shap_data = pickle.load(f)

    st.session_state["model"]         = pipeline
    st.session_state["scaler"]        = scaler
    st.session_state["lr_model"]      = lr_model
    st.session_state["X_df"]          = X_df
    st.session_state["labels_df"]     = labels_df
    st.session_state["shap_data"]     = shap_data
    st.session_state["feature_names"] = FEATURE_NAMES
    st.session_state["class_names"]   = CLASS_NAMES


# ──────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PRE-SLIP · Phenotype Screening",
    page_icon="🦴",
    layout="wide",
)

initialize_data()

# ──────────────────────────────────────────────────────────
# Main page
# ──────────────────────────────────────────────────────────
st.title("🦴 PRE-SLIP: Phenotype Screening Calculator")
st.markdown(
    "**Pre**operative **Phenotype** classification for **L**umbar **I**sthmic "
    "**S**pondylolisthesis — a data-driven screening tool using "
    "6 clinic-visit variables."
)

st.divider()

# ── Study overview ──
st.header("Study Overview")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **Background**  
    Lumbar isthmic spondylolisthesis (LIS) exhibits heterogeneous 
    spino-pelvic biomechanical profiles that may influence surgical 
    outcomes. Conventional grading (e.g. Meyerding) relies on a 
    single parameter and does not capture this heterogeneity.

    **Objective**  
    We applied unsupervised clustering to 13 spino-pelvic parameters 
    in 501 consecutive LIS patients to identify clinically meaningful 
    phenotypes, then built a simplified screening model using 6 
    variables obtainable during a routine clinic visit.
    """)

with col2:
    st.markdown("""
    **Phenotypes identified (K-Means, k = 3)**

    | Phenotype | n | Key characteristics |
    |:---|:---:|:---|
    | **Compensated** | 193 (38.5 %) | Preserved LL & SS, maintained sagittal balance |
    | **Rigid** | 129 (25.7 %) | Reduced segmental mobility, low LL |
    | **Unstable** | 179 (35.7 %) | Dynamic instability, relatively preserved SS |
    """)

st.divider()

# ── Screening model performance ──
st.header("Screening Model Performance")
st.markdown(
    "Multinomial logistic regression · 10-fold stratified cross-validation · "
    "6 predictors: **Age, BMI, Gender, VAS, LL, SS**"
)

perf = pd.read_csv(
    Path(__file__).parent / "results_v2" / "tables"
    / "T2_screening_classifier_performance.csv"
) if (Path(__file__).parent / "results_v2").exists() else None

if perf is not None:
    st.dataframe(
        perf.style.format({
            "AUC_macro": "{:.3f}", "AUC_95CI_low": "{:.3f}",
            "AUC_95CI_high": "{:.3f}", "AUC_Compensated": "{:.3f}",
            "AUC_Rigid": "{:.3f}", "AUC_Unstable": "{:.3f}",
            "Accuracy": "{:.3f}", "Macro_F1": "{:.3f}",
            "Brier_mean": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
else:
    metrics = {
        "Macro AUC": 0.881, "AUC (Compensated)": 0.915,
        "AUC (Rigid)": 0.905, "AUC (Unstable)": 0.822,
        "Accuracy": 0.743, "Macro F1": 0.741,
    }
    cols = st.columns(len(metrics))
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, f"{v:.3f}")

st.divider()

# ── Cohort summary statistics ──
st.header("Cohort Summary")

X_df = st.session_state["X_df"]
labels = st.session_state["labels_df"]

summary_rows = []
for feat in FEATURE_NAMES:
    row = {"Feature": FEATURE_DISPLAY.get(feat, feat)}
    col = X_df[feat]
    if feat == "Gender":
        row["Overall (n = 501)"] = f"{int(col.sum())} male ({col.mean()*100:.1f} %)"
        for cls in CLASS_NAMES:
            mask = labels["Phenotype"] == cls
            sub = col[mask]
            row[cls] = f"{int(sub.sum())} ({sub.mean()*100:.1f} %)"
    else:
        row["Overall (n = 501)"] = f"{col.mean():.1f} ± {col.std():.1f}"
        for cls in CLASS_NAMES:
            mask = labels["Phenotype"] == cls
            sub = col[mask]
            row[cls] = f"{sub.mean():.1f} ± {sub.std():.1f}"
    summary_rows.append(row)

st.dataframe(
    pd.DataFrame(summary_rows),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ── Navigation hint ──
st.info(
    "👈  Use the sidebar to navigate to **New Patient Prediction** "
    "to classify a new patient and view the SHAP explanation.",
    icon="ℹ️",
)

# ── Citation ──
st.header("Citation")
st.code(
    "[Authors]. Phenotype-based classification of lumbar isthmic "
    "spondylolisthesis using unsupervised clustering: a data-driven "
    "approach for preoperative screening and prognostic stratification. "
    "The Spine Journal, 2026.",
    language=None,
)

st.caption(
    "**Disclaimer** — This tool is provided for research and educational "
    "purposes only. It is not a substitute for professional clinical "
    "judgment. External validation is recommended before clinical adoption."
)
