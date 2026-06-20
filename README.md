# 🌏 AI-Based Seismic Risk Prediction System

> **College AIML Project** — Regional seismic risk classification for Japan using 25 years of USGS earthquake data.

---

## 📋 Project Overview

This project builds a complete AI pipeline to assess, classify, and explain **regional seismic risk** across Japan using historical earthquake data from the USGS catalog (2000–2025, Magnitude ≥ 4).

**It does NOT predict individual earthquakes.** It classifies regional seismic risk levels (Low / Medium / High) based on long-term patterns, and provides tools for understanding and visualizing those risks.

---

## 🚀 Live Dashboard

Run locally with:
```bash
cd AI-Based-Seismic-Risk-Prediction-System
streamlit run app.py
```
Then open **http://localhost:8501** in your browser.

---

## 🔬 Project Pipeline

| Phase | Component | Output |
|-------|-----------|--------|
| 1 | Data Collection | `cleaned_japan_earthquake_dataset.csv` |
| 2 | Data Cleaning | Filtered to Japan bbox, parsed dates |
| 3 | Feature Engineering | `engineered_seismic_features.csv` |
| 4 | Risk Label Generation | `labeled_seismic_features.csv` |
| 5 | **Random Forest** | `models/random_forest_model.joblib` — 70.55% accuracy |
| 6 | **K-Means Clustering** | `seismic_regions_clusters.csv` — 4 seismic zones |
| 7 | **Isolation Forest** | `seismic_anomalies.csv` — 82 anomalous region-years |
| 8 | **SHAP Explainability** | `shap_global_importance.csv`, `shap_summary_plot.png` |
| 9 | **H3 Hexagonal Mapping** | `h3_risk_map.csv` — Uber H3 resolution 5 cells |
| 10 | **A\* Route Planner** | Risk-weighted safe evacuation path finding |
| 11 | **Streamlit Dashboard** | `app.py` — 7-page interactive dashboard |

---

## 📊 Model Results

| Metric | Value |
|--------|-------|
| Random Forest Test Accuracy | **70.55%** |
| 2020 Backtesting Accuracy | **74.16%** |
| K-Means Clusters | **4** (Quiet / Active / Deep / Tohoku hotspot) |
| Anomalies Detected (1% rate) | **82** of 8,188 region-years |
| Top SHAP Feature | `eq_count_3yr` (0.0684 mean |SHAP|) |

---

## 📂 Project Structure

```
AI-Based-Seismic-Risk-Prediction-System/
│
├── app.py                          # Streamlit dashboard (7 pages)
│
├── src/                            # Source scripts
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── label_generation.py
│   ├── train_rf.py
│   ├── run_kmeans.py
│   ├── run_isolation_forest.py
│   ├── shap_analysis.py
│   ├── run_h3_mapping.py
│   └── route_planner.py
│
├── models/                         # Trained model files
│   ├── random_forest_model.joblib
│   ├── isolation_forest_model.joblib
│   ├── kmeans_model.joblib
│   ├── scaler_kmeans.joblib
│   └── feature_names.joblib
│
├── cleaned_japan_earthquake_dataset.csv
├── engineered_seismic_features.csv
├── labeled_seismic_features.csv
├── seismic_regions_clusters.csv
├── seismic_anomalies.csv
├── h3_risk_map.csv
├── shap_global_importance.csv
├── shap_vs_rf_importance.csv
└── shap_summary_plot.png
```

---

## 🔧 Installation

```bash
pip install streamlit plotly folium streamlit-folium pandas numpy scikit-learn joblib h3 shap matplotlib
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.14 | Core language |
| scikit-learn | Random Forest, K-Means, Isolation Forest |
| SHAP 0.52 | Model explainability |
| Uber H3 | Hexagonal spatial indexing |
| Plotly | Interactive charts |
| Folium | Geographic maps |
| Streamlit | Dashboard framework |
| A\* Search | Safe evacuation route planning |

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Project Overview | Pipeline, key metrics, tech stack |
| 🎯 Risk Prediction | Confusion matrix, live prediction demo with sliders |
| 🗺️ H3 Risk Map | Interactive Folium map with hexagonal risk zones |
| 🔵 K-Means Clusters | Seismic zone clustering scatter + radar charts |
| ⚠️ Anomaly Detection | Isolation Forest anomaly map + timeline |
| 🔍 SHAP Explainability | Global/local SHAP importance charts |
| 🚨 A\* Route Planner | Safe evacuation route comparison between cities |

---

## ⚠️ Disclaimer

This system is built for **academic demonstration purposes**. It should not be used as a substitute for professional seismic hazard assessments or real-time emergency management systems.

---

## 📄 Dataset Source

- **USGS Earthquake Catalog**: [earthquake.usgs.gov](https://earthquake.usgs.gov/earthquakes/search/)
- Region: Japan (24°N–48°N, 122°E–150°E)
- Period: 2000–2025 | Magnitude ≥ 4
