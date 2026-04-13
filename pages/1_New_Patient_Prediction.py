import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ── Import initialiser from main app ──
try:
    from app import initialize_data
except ImportError:
    st.error("Could not import the main app module.")
    st.stop()

initialize_data()

# ── Guard: data must be loaded ──
if "model" not in st.session_state:
    st.error("Model not loaded. Please return to the main page first.")
    st.stop()

# ── Pull objects from session ──
pipeline      = st.session_state["model"]
scaler        = st.session_state["scaler"]
lr_model      = st.session_state["lr_model"]
X_df          = st.session_state["X_df"]
feature_names = st.session_state["feature_names"]
class_names   = st.session_state["class_names"]

FEATURE_DISPLAY = {
    "Age":    "Age",
    "BMI":    "Body Mass Index",
    "Gender": "Gender",
    "VAS":    "Visual Analogue Scale",
    "LL":     "Lumbar Lordosis (°)",
    "SS":     "Sacral Slope (°)",
}

# ──────────────────────────────────────────────────────────
st.title("🔬 New Patient Prediction")
st.markdown(
    "Enter the patient's 6 clinic-visit parameters below. "
    "The model will predict the most likely **phenotype** and "
    "display a SHAP waterfall plot explaining the contribution "
    "of each variable."
)
st.divider()

# ── Input form ──
st.header("Patient Parameters")

gender_map = {"Female": 0, "Male": 1}

input_data = {}
cols = st.columns(3)

# Row 1: Age, Gender, BMI
with cols[0]:
    input_data["Age"] = st.number_input(
        "Age (years)",
        value=int(round(X_df["Age"].mean())),
        min_value=18, max_value=100, step=1,
    )
with cols[1]:
    selected_gender = st.selectbox("Gender", options=["Female", "Male"], index=0)
    input_data["Gender"] = gender_map[selected_gender]
with cols[2]:
    input_data["BMI"] = st.number_input(
        "BMI (kg/m²)",
        value=round(float(X_df["BMI"].mean()), 1),
        min_value=10.0, max_value=60.0, step=0.1,
        format="%.1f",
    )

# Row 2: VAS, LL, SS
cols2 = st.columns(3)
with cols2[0]:
    vas_options = list(range(11))
    default_vas = int(round(X_df["VAS"].mean()))
    input_data["VAS"] = st.selectbox(
        "VAS (0–10)", options=vas_options,
        index=vas_options.index(default_vas),
    )
with cols2[1]:
    input_data["LL"] = st.number_input(
        "Lumbar Lordosis LL (°)",
        value=round(float(X_df["LL"].mean()), 1),
        min_value=0.0, max_value=120.0, step=0.1,
        format="%.1f",
    )
with cols2[2]:
    input_data["SS"] = st.number_input(
        "Sacral Slope SS (°)",
        value=round(float(X_df["SS"].mean()), 1),
        min_value=0.0, max_value=80.0, step=0.1,
        format="%.1f",
    )

st.divider()

# ── Prediction ──
if st.button("🧬 Get Prediction", type="primary", use_container_width=True):

    # Prepare input
    patient_df = pd.DataFrame(
        [[input_data[f] for f in feature_names]],
        columns=feature_names,
    )

    # Predict
    pred_idx = pipeline.predict(patient_df.values)[0]
    pred_proba = pipeline.predict_proba(patient_df.values)[0]
    pred_label = class_names[pred_idx]

    # ── Results ──
    st.header("Prediction Results")

    # Metric cards
    res_cols = st.columns([1, 3])
    with res_cols[0]:
        colour = {"Compensated": "blue", "Rigid": "red", "Unstable": "green"}
        st.metric("Predicted Phenotype", pred_label)

    with res_cols[1]:
        proba_df = pd.DataFrame(
            {"Phenotype": class_names, "Probability": pred_proba}
        ).set_index("Phenotype")
        st.dataframe(
            proba_df.style
            .format({"Probability": "{:.2%}"})
            .bar(subset=["Probability"], color="#4A90D9", vmin=0, vmax=1),
            use_container_width=True,
        )

    # Probability bar chart
    fig_bar, ax_bar = plt.subplots(figsize=(6, 2.2))
    bar_colors = ["#0072B2", "#D55E00", "#009E73"]
    bars = ax_bar.barh(class_names, pred_proba, color=bar_colors, height=0.5, edgecolor="white")
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("Predicted probability")
    for bar, p in zip(bars, pred_proba):
        ax_bar.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{p:.1%}", va="center", fontsize=9)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    fig_bar.tight_layout()
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    st.divider()

    # ── SHAP explanations ──
    st.header("Prediction Explanation (SHAP Waterfall Plots)")
    st.markdown(
        "The waterfall plots below decompose the prediction for each phenotype. "
        "**Red** features push the probability **higher**; "
        "**blue** features push it **lower**."
    )

    # Compute SHAP for this patient
    patient_scaled = scaler.transform(patient_df.values)

    explainer = shap.LinearExplainer(
        lr_model, scaler.transform(X_df.values),
        feature_names=feature_names,
    )
    shap_values_patient = explainer.shap_values(patient_scaled)

    # One waterfall per class
    shap_cols = st.columns(len(class_names))
    for i, cls in enumerate(class_names):
        with shap_cols[i]:
            st.subheader(cls)

            explanation = shap.Explanation(
                values=shap_values_patient[i][0] if isinstance(shap_values_patient, list) else shap_values_patient[0, :, i],
                base_values=explainer.expected_value[i] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value,
                data=patient_df.iloc[0].values,
                feature_names=feature_names,
            )

            fig_shap, ax_shap = plt.subplots(figsize=(5, 3.5))
            shap.plots.waterfall(explanation, max_display=6, show=False)
            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close(fig_shap)

    st.divider()
    st.caption(
        "SHAP values are computed using `shap.LinearExplainer` on the "
        "multinomial logistic regression screening model. "
        "Feature contributions are shown on the log-odds scale."
    )
