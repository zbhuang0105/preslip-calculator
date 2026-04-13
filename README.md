# PRE-SLIP: Phenotype Screening Calculator

A web-based clinical prediction tool for **Pre**operative **Phenotype** classification of **L**umbar **I**sthmic **S**pondylolisthesis.

## Overview

This Streamlit application implements a multinomial logistic regression model that classifies LIS patients into three biomechanical phenotypes using **6 clinic-visit variables**:

| Variable | Description |
|:---------|:------------|
| Age | Patient age (years) |
| BMI | Body Mass Index (kg/m²) |
| Gender | Female / Male |
| VAS | Visual Analogue Scale (0–10) |
| LL | Lumbar Lordosis (°) |
| SS | Sacral Slope (°) |

### Phenotypes

- **Compensated** — Preserved lordosis and sacral slope; maintained sagittal balance
- **Rigid** — Reduced segmental mobility; diminished lordosis
- **Unstable** — Dynamic segmental instability

### Model Performance (10-fold CV)

| Metric | Value |
|:-------|------:|
| Macro AUC | 0.881 |
| Accuracy | 74.3 % |
| Macro F1 | 0.741 |

## Deployment

### Streamlit Cloud (recommended)

1. Fork or push this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub and select this repository
4. Set **Main file path** to `app.py`
5. Click **Deploy**

### Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Citation

> [Authors]. Phenotype-based classification of lumbar isthmic spondylolisthesis using unsupervised clustering: a data-driven approach for preoperative screening and prognostic stratification. *The Spine Journal*, 2026.

## Disclaimer

This tool is provided for **research and educational purposes only**. It is not intended to substitute for professional clinical judgment. External validation is recommended before clinical adoption.
