"""
AI-Based Seismic Risk Prediction System — Streamlit Dashboard
USGS Japan Earthquake Dataset | 2000–2025 | Magnitude >= 4
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import folium
from streamlit_folium import st_folium
import h3
import math, heapq
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Seismic Risk Prediction | Japan",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; padding: 6px 0; }

/* Main bg */
.main { background-color: #0f172a; color: #e2e8f0; }
.block-container { padding: 2rem 2.5rem 2rem 2.5rem; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f2744 50%, #1a1a3e 100%);
    border: 1px solid #2563eb33;
    border-radius: 16px;
    padding: 2.5rem 2.5rem;
    margin-bottom: 2rem;
}
.hero-title { font-size: 2.4rem; font-weight: 700; color: #e2e8f0; margin: 0; }
.hero-sub { font-size: 1rem; color: #94a3b8; margin-top: 0.5rem; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #38bdf8; }
.metric-label { font-size: 0.8rem; color: #94a3b8; margin-top: 4px; letter-spacing: 0.05em; text-transform: uppercase; }

/* Risk badges */
.badge-high { background:#dc2626; color:#fff; padding:3px 12px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.badge-medium { background:#d97706; color:#fff; padding:3px 12px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.badge-low { background:#16a34a; color:#fff; padding:3px 12px; border-radius:20px; font-size:0.8rem; font-weight:600; }

/* Section headers */
.section-header {
    font-size: 1.3rem; font-weight: 700; color: #38bdf8;
    border-left: 4px solid #2563eb; padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
}

/* Info box */
.info-box {
    background: #1e3a5f22;
    border: 1px solid #2563eb44;
    border-left: 4px solid #2563eb;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.9rem;
    color: #cbd5e1;
    margin: 1rem 0;
}

/* Tables */
.dataframe { background: #1e293b !important; color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING (CACHED)
# ─────────────────────────────────────────────
BASE = os.path.dirname(__file__)

@st.cache_data
def load_data():
    h3_map = pd.read_csv(os.path.join(BASE, "h3_risk_map.csv"))
    clusters = pd.read_csv(os.path.join(BASE, "seismic_regions_clusters.csv"))
    anomalies = pd.read_csv(os.path.join(BASE, "seismic_anomalies.csv"))
    shap_df = pd.read_csv(os.path.join(BASE, "shap_global_importance.csv"))
    labeled = pd.read_csv(os.path.join(BASE, "labeled_seismic_features.csv"))
    shap_vs_rf = pd.read_csv(os.path.join(BASE, "shap_vs_rf_importance.csv"))
    return h3_map, clusters, anomalies, shap_df, labeled, shap_vs_rf

@st.cache_resource
def load_models():
    model_dir = os.path.join(BASE, "models")
    rf  = joblib.load(os.path.join(model_dir, "random_forest_model.joblib"))
    iso = joblib.load(os.path.join(model_dir, "isolation_forest_model.joblib"))
    km  = joblib.load(os.path.join(model_dir, "kmeans_model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler_kmeans.joblib"))
    feat_cols = joblib.load(os.path.join(model_dir, "feature_names.joblib"))
    return rf, iso, km, scaler, feat_cols

h3_map, clusters, anomalies, shap_df, labeled, shap_vs_rf = load_data()
rf, iso, km, scaler, feat_cols = load_models()

RISK_COLORS = {"High": "#dc2626", "Medium": "#f59e0b", "Low": "#22c55e"}
CLUSTER_NAMES = {
    0: "Quiet Crustal Zone",
    1: "Active Plate Boundary",
    2: "Deep Subduction Slab",
    3: "Tohoku Mega-Hotspot",
}

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌏 Seismic Risk AI")
    st.markdown("---")
    page = st.radio(
        "Navigate to:",
        options=[
            "🏠 Project Overview",
            "🎯 Risk Prediction",
            "🗺️ H3 Risk Map",
            "🔵 K-Means Clusters",
            "⚠️ Anomaly Detection",
            "🔍 SHAP Explainability",
            "🚨 A* Route Planner",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Dataset**: USGS Japan Earthquakes")
    st.markdown("**Period**: 2000 – 2025 | Mag ≥ 4")
    st.markdown("**Total Records**: ~29,951 events")
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#64748b;'>College AIML Project<br>AI-Based Seismic Risk System</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE A — PROJECT OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Project Overview":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🌏 AI-Based Seismic Risk Prediction System</div>
        <div class="hero-sub">
            USGS Japan Earthquake Dataset &nbsp;|&nbsp; 2000–2025 &nbsp;|&nbsp; Magnitude ≥ 4
            &nbsp;|&nbsp; Random Forest · K-Means · Isolation Forest · SHAP · H3 · A*
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-value">29,951</div><div class="metric-label">Earthquake Events</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-value">356</div><div class="metric-label">Seismic Regions</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-value">70.55%</div><div class="metric-label">RF Accuracy</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-value">74.16%</div><div class="metric-label">2020 Backtest</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="metric-card"><div class="metric-value">4</div><div class="metric-label">Seismic Clusters</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # Project pipeline diagram
    st.markdown('<div class="section-header">📋 Project Architecture Pipeline</div>', unsafe_allow_html=True)

    pipeline_steps = [
        ("1. Data Collection", "USGS earthquake catalog\n2000–2025, Mag ≥ 4", "#2563eb"),
        ("2. Data Cleaning", "Removed nulls, filtered Japan\nbounding box, parsed dates", "#7c3aed"),
        ("3. Feature Engineering", "3-yr / 1-yr rolling counts,\nenergy, depth, trends", "#0891b2"),
        ("4. Risk Label Generation", "HRS scoring → Low / Medium\n/ High thresholds (P50/P85)", "#059669"),
        ("5. Random Forest", "100 trees, depth=12\nbalanced weights", "#d97706"),
        ("6. K-Means Clustering", "K=4 clusters, spatial\nhazard zone grouping", "#dc2626"),
        ("7. Isolation Forest", "Anomaly detection on\nfeature space (1%)", "#7c3aed"),
        ("8. SHAP Explainability", "Global + local feature\nimportance analysis", "#0891b2"),
        ("9. H3 Mapping", "Hexagonal spatial index\nresolution 5 cells", "#059669"),
        ("10. A* Route Planner", "Risk-weighted graph search\nfor safe evacuation paths", "#d97706"),
    ]

    cols = st.columns(5)
    for i, (title, desc, color) in enumerate(pipeline_steps):
        with cols[i % 5]:
            st.markdown(f"""
            <div style="background:#1e293b; border:1px solid {color}44; border-top:3px solid {color};
                        border-radius:10px; padding:1rem; margin-bottom:1rem; min-height:110px;">
                <div style="font-weight:700; color:{color}; font-size:0.85rem;">{title}</div>
                <div style="font-size:0.78rem; color:#94a3b8; margin-top:6px; white-space:pre-line;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Risk label distribution
    st.markdown('<div class="section-header">📊 Risk Label Distribution (Full Dataset)</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        dist = labeled["risk_label"].value_counts().reset_index()
        dist.columns = ["Risk Level", "Count"]
        dist["Percentage"] = (dist["Count"] / dist["Count"].sum() * 100).round(1)
        fig_pie = px.pie(
            dist, values="Count", names="Risk Level",
            color="Risk Level",
            color_discrete_map=RISK_COLORS,
            hole=0.55,
        )
        fig_pie.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font_color="#e2e8f0", margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(bgcolor="#1e293b", bordercolor="#334155"),
            showlegend=True,
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        yearly = labeled.groupby(["year", "risk_label"]).size().reset_index(name="count")
        fig_bar = px.bar(
            yearly, x="year", y="count", color="risk_label",
            color_discrete_map=RISK_COLORS,
            labels={"count": "Region-Year Count", "year": "Year", "risk_label": "Risk Level"},
            title="Risk Label Distribution by Year",
            barmode="stack",
        )
        fig_bar.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            font_color="#e2e8f0", margin=dict(t=40, b=20),
            legend=dict(bgcolor="#1e293b"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Tech stack
    st.markdown('<div class="section-header">🔧 Technology Stack</div>', unsafe_allow_html=True)
    tcols = st.columns(4)
    tech = [
        ("🐍 Python 3.14", "Core language"),
        ("🌲 Scikit-Learn", "Random Forest, K-Means,\nIsolation Forest"),
        ("🧠 SHAP 0.52", "Model explainability"),
        ("🗺️ Uber H3", "Hexagonal spatial index"),
        ("📊 Plotly", "Interactive charts"),
        ("🗺️ Folium", "Geographic maps"),
        ("⚡ Streamlit", "Dashboard framework"),
        ("🔍 A* Search", "Safe route planning"),
    ]
    for i, (name, desc) in enumerate(tech):
        with tcols[i % 4]:
            st.markdown(f"""
            <div style="background:#1e293b; border:1px solid #334155; border-radius:10px;
                        padding:0.9rem 1rem; margin-bottom:0.8rem;">
                <div style="font-weight:700; color:#e2e8f0; font-size:0.9rem;">{name}</div>
                <div style="font-size:0.78rem; color:#64748b; white-space:pre-line;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE B — RISK PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🎯 Risk Prediction":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🎯 Seismic Risk Prediction</div>
        <div class="hero-sub">Random Forest Classifier — 70.55% Accuracy | 74.16% on 2020 Backtest</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "🔮 Live Prediction Demo", "📈 Feature Importance"])

    with tab1:
        st.markdown('<div class="section-header">Confusion Matrix (Test Set 2020–2025)</div>', unsafe_allow_html=True)

        # Compute confusion matrix on test set
        test_df = labeled[labeled["year"] >= 2019]
        X_test = test_df[feat_cols]
        y_test = test_df["risk_label"]
        y_pred = rf.predict(X_test)

        classes = ["Low", "Medium", "High"]
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        cr = classification_report(y_test, y_pred, labels=classes, output_dict=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            fig_cm = px.imshow(
                cm, x=classes, y=classes,
                text_auto=True, aspect="auto",
                color_continuous_scale="Blues",
                labels={"x": "Predicted", "y": "Actual", "color": "Count"},
                title="Confusion Matrix (Test Set)",
            )
            fig_cm.update_layout(
                paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                font_color="#e2e8f0", margin=dict(t=50, b=20),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
            cr_rows = []
            for cls in classes:
                cr_rows.append({
                    "Class": cls,
                    "Precision": f"{cr[cls]['precision']:.2%}",
                    "Recall": f"{cr[cls]['recall']:.2%}",
                    "F1-Score": f"{cr[cls]['f1-score']:.2%}",
                    "Support": int(cr[cls]["support"]),
                })
            st.dataframe(pd.DataFrame(cr_rows), use_container_width=True, hide_index=True)

            st.markdown('<div class="section-header">Overall Metrics</div>', unsafe_allow_html=True)
            mc1, mc2 = st.columns(2)
            mc1.metric("Test Accuracy", "70.55%")
            mc2.metric("2020 Backtest", "74.16%")

        # Year-by-year accuracy
        st.markdown('<div class="section-header">Year-by-Year Accuracy (Test Period)</div>', unsafe_allow_html=True)
        year_acc = []
        for yr in sorted(test_df["year"].unique()):
            sub = test_df[test_df["year"] == yr]
            acc = (rf.predict(sub[feat_cols]) == sub["risk_label"]).mean()
            year_acc.append({"Year": yr, "Accuracy": acc})
        year_acc_df = pd.DataFrame(year_acc)
        fig_acc = px.bar(
            year_acc_df, x="Year", y="Accuracy",
            text_auto=".1%",
            color="Accuracy", color_continuous_scale=["#dc2626", "#f59e0b", "#22c55e"],
            title="Accuracy per Test Year",
        )
        fig_acc.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            font_color="#e2e8f0", yaxis_tickformat=".0%",
            showlegend=False, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-header">🔮 Live Prediction Tool</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Adjust the sliders below to simulate seismic features for any region-year, then run the model to get a predicted risk classification.</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            grid_lat = st.slider("Grid Latitude (°N)", 24, 46, 36)
            grid_lon = st.slider("Grid Longitude (°E)", 122, 149, 141)
            year_val = st.slider("Year", 2000, 2024, 2023)
        with col2:
            eq_count_3yr = st.slider("Eq Count (3-Year Window)", 0, 200, 30)
            eq_count_1yr = st.slider("Eq Count (1-Year Window)", 0, 100, 10)
            freq_trend = st.slider("Frequency Trend", -5.0, 30.0, 1.5)
        with col3:
            mean_mag_3yr = st.slider("Mean Magnitude (3yr)", 4.0, 7.0, 4.5)
            max_mag_3yr  = st.slider("Max Magnitude (3yr)", 4.0, 9.0, 5.5)
            mean_depth_3yr = st.slider("Mean Depth km (3yr)", 0, 250, 30)
            seismic_energy_3yr = st.number_input("Seismic Energy 3yr (J)", value=5e8, format="%.2e")
            seismic_energy_1yr = st.number_input("Seismic Energy 1yr (J)", value=1e8, format="%.2e")
            mag_trend = st.slider("Magnitude Trend", -2.0, 2.0, 0.1)

        if st.button("🔮 Predict Risk Level", type="primary"):
            input_data = pd.DataFrame([{
                "grid_lat": grid_lat, "grid_lon": grid_lon, "year": year_val,
                "eq_count_3yr": eq_count_3yr, "eq_count_1yr": eq_count_1yr,
                "mean_mag_3yr": mean_mag_3yr, "max_mag_3yr": max_mag_3yr,
                "mean_depth_3yr": mean_depth_3yr,
                "seismic_energy_3yr": seismic_energy_3yr,
                "seismic_energy_1yr": seismic_energy_1yr,
                "freq_trend": freq_trend, "mag_trend": mag_trend,
            }])

            pred = rf.predict(input_data[feat_cols])[0]
            probs = rf.predict_proba(input_data[feat_cols])[0]
            classes_order = list(rf.classes_)

            badge_map = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}
            icon_map = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

            st.markdown(f"""
            <div style="background:#1e293b; border:2px solid {RISK_COLORS.get(pred,'#334155')};
                        border-radius:14px; padding:1.5rem 2rem; margin-top:1rem; text-align:center;">
                <div style="font-size:3rem;">{icon_map.get(pred,'❓')}</div>
                <div style="font-size:1.8rem; font-weight:700; color:{RISK_COLORS.get(pred,'#e2e8f0')}; margin-top:0.5rem;">
                    {pred} Risk
                </div>
                <div style="font-size:0.9rem; color:#94a3b8; margin-top:0.5rem;">
                    Region: {grid_lat}°N, {grid_lon}°E | Year: {year_val}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            prob_cols = st.columns(3)
            for i, cls in enumerate(["High", "Medium", "Low"]):
                idx = classes_order.index(cls)
                prob_cols[i].metric(f"{icon_map[cls]} {cls} Risk", f"{probs[idx]:.1%}")

    with tab3:
        st.markdown('<div class="section-header">Random Forest Feature Importance (Gini)</div>', unsafe_allow_html=True)
        imp_df = pd.DataFrame({"Feature": feat_cols, "Importance": rf.feature_importances_})
        imp_df = imp_df.sort_values("Importance", ascending=True)
        fig_imp = px.bar(
            imp_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale=["#1e3a5f", "#2563eb", "#38bdf8"],
            title="Gini Feature Importance (Random Forest)",
        )
        fig_imp.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            font_color="#e2e8f0", showlegend=False,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE C — H3 RISK MAP
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🗺️ H3 Risk Map":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🗺️ H3 Hexagonal Risk Map</div>
        <div class="hero-sub">Uber H3 Resolution 5 | Each hexagon ≈ 2,525 km² | Color-coded by predicted risk</div>
    </div>
    """, unsafe_allow_html=True)

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        year_sel = st.selectbox("Select Year", sorted(h3_map["year"].unique(), reverse=True), key="h3_year")
    with col_f2:
        risk_filter = st.multiselect(
            "Show Risk Levels",
            ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
            key="h3_risk",
        )

    df_plot = h3_map[(h3_map["year"] == year_sel) & (h3_map["predicted_risk_label"].isin(risk_filter))]

    # Build Folium map
    m = folium.Map(location=[37.5, 137.5], zoom_start=5, tiles="CartoDB dark_matter")

    color_map = {"High": "#dc2626", "Medium": "#f59e0b", "Low": "#22c55e"}

    for _, row in df_plot.iterrows():
        cell = row["h3_cell"]
        try:
            boundary = h3.cell_to_boundary(cell)
            risk = row["predicted_risk_label"]
            color = color_map.get(risk, "#64748b")
            popup_html = f"""
            <div style='font-family:sans-serif; font-size:12px; min-width:200px'>
                <b style='color:{color}'>⬡ {risk} Risk</b><br>
                <b>H3 Cell:</b> {cell}<br>
                <b>Location:</b> {row['grid_lat']}°N, {row['grid_lon']}°E<br>
                <b>Year:</b> {int(row['year'])}<br>
                <b>Prob(High):</b> {row['predicted_prob_High']:.1%}<br>
                <b>3yr Eq Count:</b> {int(row['eq_count_3yr'])}<br>
                <b>3yr Energy:</b> {row['seismic_energy_3yr']:.2e} J<br>
                <b>Mean Depth:</b> {row['mean_depth_3yr']:.1f} km
            </div>"""
            folium.Polygon(
                locations=[[lat, lon] for lat, lon in boundary],
                color=color, fill_color=color,
                fill_opacity=0.55, weight=1,
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(m)
        except Exception:
            pass

    # Legend
    legend = """<div style='position:fixed; bottom:30px; right:30px; z-index:999;
        background:#1e293b; border:1px solid #334155; border-radius:10px;
        padding:12px 18px; font-family:sans-serif; font-size:13px; color:#e2e8f0;'>
        <b>Risk Level</b><br>
        <span style='color:#dc2626'>■</span> High Risk<br>
        <span style='color:#f59e0b'>■</span> Medium Risk<br>
        <span style='color:#22c55e'>■</span> Low Risk
        </div>"""
    m.get_root().html.add_child(folium.Element(legend))

    st_folium(m, height=550, use_container_width=True)

    # Summary table
    st.markdown('<div class="section-header">Risk Summary Table</div>', unsafe_allow_html=True)
    summary = df_plot.groupby("predicted_risk_label").agg(
        Cells=("h3_cell", "count"),
        Avg_Prob_High=("predicted_prob_High", "mean"),
        Avg_3yr_Count=("eq_count_3yr", "mean"),
        Avg_Depth=("mean_depth_3yr", "mean"),
    ).reset_index().rename(columns={"predicted_risk_label": "Risk Level"})
    st.dataframe(summary.style.format({
        "Avg_Prob_High": "{:.1%}",
        "Avg_3yr_Count": "{:.1f}",
        "Avg_Depth": "{:.1f} km",
    }), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE D — K-MEANS CLUSTERS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔵 K-Means Clusters":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🔵 K-Means Seismic Zone Clustering</div>
        <div class="hero-sub">K=4 clusters identify distinct seismic hazard profiles across Japan's tectonic regions</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🗺️ Cluster Map", "📊 Cluster Profiles"])

    with tab1:
        cluster_colors = {0: "#38bdf8", 1: "#f59e0b", 2: "#a78bfa", 3: "#dc2626"}
        fig_cl = go.Figure()
        for cid, cname in CLUSTER_NAMES.items():
            sub = clusters[clusters["cluster"] == cid]
            fig_cl.add_trace(go.Scatter(
                x=sub["grid_lon"], y=sub["grid_lat"],
                mode="markers",
                marker=dict(
                    size=10,
                    color=cluster_colors[cid],
                    symbol="hexagon",
                    opacity=0.85,
                    line=dict(color="#0f172a", width=0.5),
                ),
                name=f"Cluster {cid}: {cname}",
                text=[f"{cname}<br>Lat:{lat}N Lon:{lon}E<br>Avg Count:{cnt:.1f}<br>Max Mag:{mag:.1f}"
                      for lat, lon, cnt, mag in zip(sub.grid_lat, sub.grid_lon, sub.avg_annual_count, sub.max_magnitude)],
                hovertemplate="%{text}<extra></extra>",
            ))
        fig_cl.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            font_color="#e2e8f0",
            xaxis_title="Longitude (°E)",
            yaxis_title="Latitude (°N)",
            title="K-Means Cluster Map of Japan's Seismic Zones",
            legend=dict(bgcolor="#1e293b", bordercolor="#334155"),
            margin=dict(t=50, b=20),
            height=500,
        )
        st.plotly_chart(fig_cl, use_container_width=True)

        st.markdown('<div class="info-box">Each hexagon marker represents a 1°×1° grid cell. Color indicates the K-Means cluster assigned based on long-term seismic activity features (earthquake frequency, max magnitude, average depth, energy release).</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-header">Cluster Characteristic Profiles</div>', unsafe_allow_html=True)

        cluster_desc = {
            0: ("🔵 Quiet Crustal Zone", "#38bdf8",
                "Low seismicity. Inland and remote regions with minimal tectonic loading. Low earthquake frequency and shallow depths."),
            1: ("🟡 Active Plate Boundary", "#f59e0b",
                "Moderate to high seismicity. Regions near subduction interfaces. High frequency with moderate magnitude events."),
            2: ("🟣 Deep Subduction Slab", "#a78bfa",
                "Intermediate depth seismicity. Deep events from subducting oceanic slab, lower surface hazard than shallow quakes."),
            3: ("🔴 Tohoku Mega-Hotspot", "#dc2626",
                "Extreme seismicity. Represents the Tohoku trench rupture zone (2011 M9.0 epicenter). Highest frequency and energy."),
        }
        for cid, (name, color, desc) in cluster_desc.items():
            sub = clusters[clusters["cluster"] == cid]
            st.markdown(f"""
            <div style="background:#1e293b; border-left:4px solid {color}; border-radius:8px;
                        padding:1rem 1.5rem; margin-bottom:1rem;">
                <div style="font-weight:700; color:{color}; font-size:1rem;">{name}</div>
                <div style="color:#94a3b8; font-size:0.85rem; margin:6px 0;">{desc}</div>
                <div style="display:flex; gap:2rem; font-size:0.85rem; color:#e2e8f0; margin-top:8px;">
                    <span>📍 <b>{len(sub)}</b> cells</span>
                    <span>⚡ Avg Count: <b>{sub.avg_annual_count.mean():.1f}/yr</b></span>
                    <span>🔴 Max Mag: <b>{sub.max_magnitude.max():.1f}</b></span>
                    <span>📏 Avg Depth: <b>{sub.avg_depth.mean():.1f} km</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Cluster Feature Comparison</div>', unsafe_allow_html=True)
        cluster_stats = clusters.groupby("cluster").agg(
            Cells=("grid_lat", "count"),
            Avg_Count=("avg_annual_count", "mean"),
            Max_Magnitude=("max_magnitude", "max"),
            Avg_Magnitude=("avg_magnitude", "mean"),
            Avg_Depth=("avg_depth", "mean"),
        ).reset_index()
        cluster_stats["Cluster Name"] = cluster_stats["cluster"].map(CLUSTER_NAMES)
        fig_radar = go.Figure()
        cats = ["Avg_Count", "Max_Magnitude", "Avg_Magnitude", "Avg_Depth"]
        for _, row in cluster_stats.iterrows():
            cid = int(row["cluster"])
            vals = [row[c] for c in cats]
            norm_vals = [v / cluster_stats[c].max() for v, c in zip(vals, cats)]
            norm_vals.append(norm_vals[0])
            fig_radar.add_trace(go.Scatterpolar(
                r=norm_vals,
                theta=cats + [cats[0]],
                fill="toself",
                name=f"C{cid}: {CLUSTER_NAMES[cid]}",
                line_color=list(cluster_colors.values())[cid],
                fillcolor=list(cluster_colors.values())[cid],
                opacity=0.4,
            ))
        fig_radar.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font_color="#e2e8f0",
            polar=dict(
                bgcolor="#1e293b",
                radialaxis=dict(visible=True, range=[0, 1], color="#64748b"),
                angularaxis=dict(color="#94a3b8"),
            ),
            legend=dict(bgcolor="#1e293b"),
            title="Cluster Feature Profiles (Normalized)",
            margin=dict(t=50),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE E — ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚠️ Anomaly Detection":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">⚠️ Isolation Forest Anomaly Detection</div>
        <div class="hero-sub">Identifies unusual seismic behaviour compared to long-term historical patterns</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="info-box">⚠️ <b>Important:</b> Anomalies represent unusual seismic behaviour patterns. They do NOT directly predict earthquakes. They indicate region-year combinations that deviate significantly from long-term expected patterns.</div>', unsafe_allow_html=True)

    detected = anomalies[anomalies["anomaly_flag"] == -1]
    normal   = anomalies[anomalies["anomaly_flag"] ==  1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Region-Years", f"{len(anomalies):,}")
    c2.metric("Anomalous", f"{len(detected):,}", delta=f"{len(detected)/len(anomalies):.1%}")
    c3.metric("Normal", f"{len(normal):,}")
    c4.metric("Detection Rate", "1%", delta="Contamination threshold")

    tab1, tab2, tab3 = st.tabs(["🗺️ Anomaly Map", "📅 Timeline", "📋 Top Anomalies"])

    with tab1:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            year_anom = st.selectbox("Year", sorted(anomalies["year"].unique(), reverse=True), key="anom_year")
        with col_f2:
            show_all = st.checkbox("Show all points (including normal)", value=False)

        df_anom_year = anomalies[anomalies["year"] == year_anom]
        if not show_all:
            df_anom_year = df_anom_year[df_anom_year["anomaly_flag"] == -1]

        fig_anom = go.Figure()
        if show_all:
            norm_yr = df_anom_year[df_anom_year["anomaly_flag"] == 1]
            fig_anom.add_trace(go.Scatter(
                x=norm_yr["grid_lon"], y=norm_yr["grid_lat"],
                mode="markers",
                marker=dict(size=7, color="#334155", opacity=0.5),
                name="Normal",
            ))
        anom_yr = df_anom_year[df_anom_year["anomaly_flag"] == -1] if show_all else df_anom_year
        if not anom_yr.empty:
            fig_anom.add_trace(go.Scatter(
                x=anom_yr["grid_lon"], y=anom_yr["grid_lat"],
                mode="markers",
                marker=dict(size=14, color="#dc2626", symbol="x", line=dict(width=2)),
                name="Anomaly",
                text=[f"Lat:{la}N Lon:{lo}E<br>Score:{sc:.3f}<br>3yr Count:{cnt:.0f}<br>Max Mag:{mag:.1f}"
                      for la, lo, sc, cnt, mag in zip(
                          anom_yr.grid_lat, anom_yr.grid_lon,
                          anom_yr.anomaly_score, anom_yr.eq_count_3yr, anom_yr.max_mag_3yr)],
                hovertemplate="%{text}<extra></extra>",
            ))
        fig_anom.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            font_color="#e2e8f0",
            xaxis_title="Longitude (°E)", yaxis_title="Latitude (°N)",
            title=f"Seismic Anomalies — {year_anom}",
            legend=dict(bgcolor="#1e293b"),
            margin=dict(t=50, b=20), height=480,
        )
        st.plotly_chart(fig_anom, use_container_width=True)

    with tab2:
        anom_by_year = detected.groupby("year").size().reset_index(name="anomaly_count")
        fig_tl = px.bar(
            anom_by_year, x="year", y="anomaly_count",
            color="anomaly_count",
            color_continuous_scale=["#1e293b", "#dc2626"],
            title="Number of Seismic Anomalies Detected Per Year",
            labels={"anomaly_count": "Anomaly Count", "year": "Year"},
        )
        fig_tl.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            font_color="#e2e8f0", showlegend=False,
            margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig_tl, use_container_width=True)

    with tab3:
        top_anom = detected.nsmallest(20, "anomaly_score")[
            ["grid_lat", "grid_lon", "year", "anomaly_score",
             "eq_count_3yr", "max_mag_3yr", "mean_depth_3yr"]
        ].reset_index(drop=True)
        top_anom.index += 1
        st.dataframe(top_anom.style.format({
            "anomaly_score": "{:.4f}",
            "eq_count_3yr": "{:.0f}",
            "max_mag_3yr": "{:.1f}",
            "mean_depth_3yr": "{:.1f}",
        }), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE F — SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 SHAP Explainability":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🔍 SHAP Model Explainability</div>
        <div class="hero-sub">SHapley Additive exPlanations — Global & Local feature importance for the Random Forest model</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🌐 Global Importance", "⚖️ SHAP vs RF Comparison"])

    with tab1:
        st.markdown('<div class="section-header">Global SHAP Feature Importance</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Computed on 300 test samples using TreeExplainer. Shows the average absolute contribution of each feature across all three risk classes.</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1.5, 1])
        with col1:
            shap_sorted = shap_df.sort_values("mean_abs_shap", ascending=True)
            fig_shap = go.Figure()
            for col_name, cls_name, color in [
                ("mean_abs_shap_High", "High Risk", "#dc2626"),
                ("mean_abs_shap_Medium", "Medium Risk", "#f59e0b"),
                ("mean_abs_shap_Low", "Low Risk", "#22c55e"),
            ]:
                fig_shap.add_trace(go.Bar(
                    y=shap_sorted["feature"],
                    x=shap_sorted[col_name],
                    name=cls_name,
                    orientation="h",
                    marker_color=color,
                    opacity=0.85,
                ))
            fig_shap.update_layout(
                barmode="stack",
                paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                font_color="#e2e8f0",
                title="SHAP Mean |value| by Feature and Risk Class",
                xaxis_title="Mean |SHAP value|",
                legend=dict(bgcolor="#1e293b"),
                margin=dict(t=50, b=20),
                height=440,
            )
            st.plotly_chart(fig_shap, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">Feature Rankings</div>', unsafe_allow_html=True)
            rank_df = shap_df[["feature", "mean_abs_shap"]].sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
            rank_df.index += 1
            rank_df.columns = ["Feature", "Mean |SHAP|"]
            st.dataframe(rank_df.style.format({"Mean |SHAP|": "{:.4f}"}), use_container_width=True)

        # SHAP Summary Plot (pre-generated)
        shap_plot_path = os.path.join(BASE, "shap_summary_plot.png")
        if os.path.exists(shap_plot_path):
            st.markdown('<div class="section-header">SHAP Summary Plot</div>', unsafe_allow_html=True)
            st.image(shap_plot_path, caption="SHAP Multi-Class Summary Plot (300 Test Samples)", use_container_width=True)

    with tab2:
        st.markdown('<div class="section-header">SHAP vs. Random Forest Gini Importance Comparison</div>', unsafe_allow_html=True)

        fig_comp = go.Figure()
        comp_sorted = shap_vs_rf.sort_values("mean_abs_shap", ascending=True)
        fig_comp.add_trace(go.Bar(
            y=comp_sorted["feature"], x=comp_sorted["rf_gini_importance"],
            name="RF Gini Importance", orientation="h",
            marker_color="#2563eb", opacity=0.85,
        ))
        fig_comp.add_trace(go.Bar(
            y=comp_sorted["feature"], x=comp_sorted["mean_abs_shap"],
            name="SHAP Mean |value|", orientation="h",
            marker_color="#38bdf8", opacity=0.85,
        ))
        fig_comp.update_layout(
            barmode="group",
            paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            font_color="#e2e8f0",
            title="SHAP vs. RF Gini Importance Comparison",
            legend=dict(bgcolor="#1e293b"),
            margin=dict(t=50, b=20), height=460,
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown('<div class="section-header">Rank Comparison Table</div>', unsafe_allow_html=True)
        rank_table = shap_vs_rf[["feature", "rf_gini_importance", "mean_abs_shap", "rf_rank", "shap_rank"]].copy()
        rank_table["Rank Change"] = rank_table["rf_rank"] - rank_table["shap_rank"]
        rank_table = rank_table.sort_values("shap_rank")
        rank_table.columns = ["Feature", "RF Gini", "SHAP |value|", "RF Rank", "SHAP Rank", "Rank Change"]
        st.dataframe(rank_table.style.format({
            "RF Gini": "{:.4f}", "SHAP |value|": "{:.4f}",
        }), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE G — A* ROUTE PLANNER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🚨 A* Route Planner":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🚨 A* Safe Evacuation Route Planner</div>
        <div class="hero-sub">Risk-weighted pathfinding on H3 hexagonal grid | Low=1, Medium=5, High=20 traversal cost</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="info-box">⚠️ <b>Disclaimer:</b> This tool demonstrates AI pathfinding concepts using historical seismic risk data. It is <b>not</b> intended as a real emergency evacuation system. Real disaster routing requires live sensor data, infrastructure status, and government coordination.</div>', unsafe_allow_html=True)

    # Load risk db for selected year
    route_year = st.selectbox("Risk data year for route planning:", sorted(h3_map["year"].unique(), reverse=True))
    df_year = h3_map[h3_map["year"] == route_year]
    risk_db = {row["h3_cell"]: row["predicted_risk_label"] for _, row in df_year.iterrows()}

    # Helper functions
    def _haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = math.sin(dp/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    def _cost(cell, mode):
        if mode == "shortest":
            return 1.0
        r = risk_db.get(cell, "Low")
        return {"High": 20.0, "Medium": 5.0, "Low": 1.0}.get(r, 1.0)

    def find_route(slat, slon, elat, elon, mode):
        sc = h3.latlng_to_cell(slat, slon, 5)
        ec = h3.latlng_to_cell(elat, elon, 5)
        if sc == ec:
            return [sc], 0.0, 0.0
        sl, el = h3.cell_to_latlng(sc), h3.cell_to_latlng(ec)
        h0 = _haversine(sl[0], sl[1], el[0], el[1])
        pq = [(h0, 0.0, sc, [sc])]
        g_scores = {sc: 0.0}
        while pq:
            f, g, curr, path = heapq.heappop(pq)
            if curr == ec:
                d = sum(_haversine(*h3.cell_to_latlng(path[i]), *h3.cell_to_latlng(path[i+1]))
                        for i in range(len(path)-1))
                return path, g, d
            for nb in set(h3.grid_disk(curr, 1)) - {curr}:
                lat, lon = h3.cell_to_latlng(nb)
                if not (20 <= lat <= 48 and 120 <= lon <= 150):
                    continue
                sd = _haversine(*h3.cell_to_latlng(curr), lat, lon)
                tg = g + sd * _cost(nb, mode)
                if nb not in g_scores or tg < g_scores[nb]:
                    g_scores[nb] = tg
                    hv = _haversine(lat, lon, el[0], el[1])
                    heapq.heappush(pq, (tg + hv, tg, nb, path + [nb]))
        return None, float("inf"), 0.0

    # Preset locations
    LOCATIONS = {
        "Tokyo (Shinjuku)": (35.6895, 139.6917),
        "Sendai": (38.2682, 140.8694),
        "Osaka": (34.6937, 135.5023),
        "Sapporo (Hokkaido)": (43.0618, 141.3545),
        "Nagoya": (35.1815, 136.9066),
        "Fukushima": (37.7503, 140.4676),
        "Noto Peninsula": (37.2959, 136.9100),
        "Hiroshima": (34.3853, 132.4553),
    }

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🟢 Start Location**")
        start_name = st.selectbox("Start city:", list(LOCATIONS.keys()), index=0)
        slat, slon = LOCATIONS[start_name]
    with col2:
        st.markdown("**🔴 End Location**")
        end_name = st.selectbox("End city:", list(LOCATIONS.keys()), index=1)
        elat, elon = LOCATIONS[end_name]

    if st.button("🚨 Find Evacuation Routes", type="primary") and start_name != end_name:
        with st.spinner("Running A* Search..."):
            path_s, cost_s, dist_s = find_route(slat, slon, elat, elon, "shortest")
            path_r, cost_r, dist_r = find_route(slat, slon, elat, elon, "safest")

        if not path_s or not path_r:
            st.error("Could not find a valid route. Try different cities.")
        else:
            risks_s = [risk_db.get(c, "Low") for c in path_s]
            risks_r = [risk_db.get(c, "Low") for c in path_r]

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Shortest Distance", f"{dist_s:.1f} km")
            m2.metric("Safest Distance", f"{dist_r:.1f} km", delta=f"+{dist_r - dist_s:.1f} km detour")
            m3.metric("Shortest High-Risk Cells", risks_s.count("High"), delta_color="inverse")
            m4.metric("Safest High-Risk Cells", risks_r.count("High"), delta_color="inverse")

            # Map
            m = folium.Map(location=[(slat+elat)/2, (slon+elon)/2], zoom_start=6, tiles="CartoDB dark_matter")

            # Draw background risk hexagons
            for _, row in df_year.iterrows():
                try:
                    bnd = h3.cell_to_boundary(row["h3_cell"])
                    rc = {"High": "#dc262633", "Medium": "#f59e0b22", "Low": "#22c55e11"}.get(row["predicted_risk_label"], "#33415511")
                    folium.Polygon(
                        locations=[[la, lo] for la, lo in bnd],
                        color=rc, fill_color=rc, fill_opacity=0.4, weight=0.5,
                    ).add_to(m)
                except Exception:
                    pass

            # Draw shortest path
            if path_s:
                coords_s = [list(h3.cell_to_latlng(c)) for c in path_s]
                folium.PolyLine(coords_s, color="#3b82f6", weight=4, opacity=0.9,
                                tooltip="Shortest Distance Path").add_to(m)

            # Draw safest path
            if path_r:
                coords_r = [list(h3.cell_to_latlng(c)) for c in path_r]
                folium.PolyLine(coords_r, color="#22c55e", weight=4, opacity=0.9, dash_array="8 4",
                                tooltip="Safest Risk-Averse Path").add_to(m)

            # Markers
            folium.Marker([slat, slon], tooltip=f"🟢 START: {start_name}",
                icon=folium.Icon(color="green", icon="play")).add_to(m)
            folium.Marker([elat, elon], tooltip=f"🔴 END: {end_name}",
                icon=folium.Icon(color="red", icon="stop")).add_to(m)

            st_folium(m, height=520, use_container_width=True)

            # Risk comparison
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="section-header">🔵 Shortest Distance Path</div>', unsafe_allow_html=True)
                cnt_s = {"High": risks_s.count("High"), "Medium": risks_s.count("Medium"), "Low": risks_s.count("Low")}
                fig_s = px.bar(
                    x=list(cnt_s.keys()), y=list(cnt_s.values()),
                    color=list(cnt_s.keys()), color_discrete_map=RISK_COLORS,
                    title=f"Risk Cells | {len(path_s)} hops | {dist_s:.1f} km",
                    labels={"x": "Risk Level", "y": "Hex Cells"},
                )
                fig_s.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                    font_color="#e2e8f0", showlegend=False, margin=dict(t=50,b=20))
                st.plotly_chart(fig_s, use_container_width=True)

            with col2:
                st.markdown('<div class="section-header">🟢 Safest Risk-Averse Path</div>', unsafe_allow_html=True)
                cnt_r = {"High": risks_r.count("High"), "Medium": risks_r.count("Medium"), "Low": risks_r.count("Low")}
                fig_r = px.bar(
                    x=list(cnt_r.keys()), y=list(cnt_r.values()),
                    color=list(cnt_r.keys()), color_discrete_map=RISK_COLORS,
                    title=f"Risk Cells | {len(path_r)} hops | {dist_r:.1f} km",
                    labels={"x": "Risk Level", "y": "Hex Cells"},
                )
                fig_r.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                    font_color="#e2e8f0", showlegend=False, margin=dict(t=50,b=20))
                st.plotly_chart(fig_r, use_container_width=True)

            # Explanation
            high_avoided = risks_s.count("High") - risks_r.count("High")
            med_avoided  = risks_s.count("Medium") - risks_r.count("Medium")
            extra_km = dist_r - dist_s
            if high_avoided > 0 or med_avoided > 0:
                st.success(f"✅ **Route planner success!** The A* safest path avoided {high_avoided} High-Risk and {med_avoided} Medium-Risk cell(s) at a cost of only {extra_km:.1f} extra km.")
            else:
                st.info("ℹ️ The shortest path is already the safest for this origin-destination pair — no high-risk zones lie directly on the route.")
    elif start_name == end_name:
        st.warning("⚠️ Please select different start and end locations.")
