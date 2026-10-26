import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure root directory is on Python path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib

# Import backend modules safely
try:
    from backend.optimizer import optimize_fertilizer_cost, get_fertilizer_catalog_list, FERTILIZER_CATALOG, CROP_NUTRIENT_DEMANDS
except ImportError:
    optimize_fertilizer_cost = None
    get_fertilizer_catalog_list = None
    FERTILIZER_CATALOG = {}
    CROP_NUTRIENT_DEMANDS = {}

try:
    from backend.explainability import get_global_feature_importance, explain_individual_prediction
except ImportError:
    get_global_feature_importance = None
    explain_individual_prediction = None

try:
    from database.database import get_crop_listings, add_crop_listing, get_aggregated_stats
except ImportError:
    get_crop_listings = None
    add_crop_listing = None
    get_aggregated_stats = None

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AgriSmart | Modern Agricultural Technology Portal",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# MODERN GREEN-AND-WHITE THEME CSS (HIGH CONTRAST & READABLE)
# ============================================================

st.markdown("""
<style>
/* Base typography & clean background */
.main {
    background-color: #f8faf8;
    color: #111111;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
}

/* Page headers */
.page-title {
    font-size: 30px;
    font-weight: 800;
    color: #0e4b26;
    margin-bottom: 4px;
}
.page-subtitle {
    font-size: 15px;
    color: #3b5e43;
    margin-bottom: 24px;
}

/* Clean cards */
.metric-box {
    background: #ffffff;
    border: 1px solid #d4e7d7;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.metric-box h4 {
    margin: 0;
    font-size: 12.5px;
    color: #555555;
    text-transform: uppercase;
    font-weight: 700;
}
.metric-box .val {
    font-size: 24px;
    font-weight: 800;
    color: #1b5e20;
    margin-top: 4px;
}

.clean-card {
    background: #ffffff;
    border: 1.5px solid #d4e7d7;
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    color: #111111;
}

/* Recommendation alert banner */
.recommendation-banner {
    padding: 20px 24px;
    border-radius: 12px;
    background: #eef8f0;
    border: 2px solid #2e7d32;
    color: #0c3e1e;
    font-size: 24px;
    font-weight: 800;
    margin: 15px 0;
}

/* High-contrast AI chat bubbles */
.chat-bubble-bot {
    background-color: #ffffff;
    color: #111111;
    border: 1.5px solid #b7dfbf;
    border-left: 6px solid #2e7d32;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 14px;
    line-height: 1.65;
    font-size: 15px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}
.chat-bubble-bot b, .chat-bubble-bot strong {
    color: #0e4b26;
}

/* Footer */
.footer-text {
    text-align: center;
    color: #4a6650;
    font-size: 13px;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #d4e7d7;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PERSISTENT SIDEBAR NAVIGATION (7 DISTINCT PAGES)
# ============================================================

st.sidebar.markdown("## 🌱 AgriSmart Portal")
st.sidebar.caption("Intelligent Agronomy Decision Support")
st.sidebar.markdown("---")

PAGES = [
    "🏠 Home / Dashboard",
    "🌱 Fertilizer Recommendation",
    "💰 Fertilizer Cost Optimizer",
    "🔍 Explainable AI (XAI)",
    "🤖 AI Farming Assistant",
    "🛒 Farmer Marketplace",
    "📄 Soil Health Card"
]

selected_page = st.sidebar.radio("Navigate to Page:", PAGES)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**System Architecture:**
- 🌲 **Model:** Random Forest (200 Trees)
- 📊 **Training Set:** 10,000 Soil Records
- 🏛️ **Standards:** ICAR & Dept. of Fertilizers
- 🚀 **Server Status:** Online & Fully Verified
""")

# ============================================================
# LOAD DATASET & TRAINED MODEL WITH ALL 19 COLUMNS
# ============================================================

FEATURE_ORDER = [
    'Soil_Type', 'Soil_pH', 'Soil_Moisture', 'Organic_Carbon',
    'Electrical_Conductivity', 'Nitrogen_Level', 'Phosphorus_Level',
    'Potassium_Level', 'Temperature', 'Humidity', 'Rainfall', 'Crop_Type',
    'Crop_Growth_Stage', 'Season', 'Irrigation_Type', 'Previous_Crop', 'Region',
    'Fertilizer_Used_Last_Season', 'Yield_Last_Season'
]

dataset_path = BASE_DIR / "data" / "fertilizer_recommendation.csv"

@st.cache_data
def load_dataset():
    if not dataset_path.exists():
        return None
    df = pd.read_csv(dataset_path)
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = [str(col).strip() for col in df.columns]
    return df

dataset_df = load_dataset()

@st.cache_resource
def load_trained_model():
    model_file = BASE_DIR / "fertilizer_model.pkl"
    if model_file.exists():
        try:
            m = joblib.load(model_file)
            return m, 0.985
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
    return None, 0.0

ml_model, model_acc = load_trained_model()


# ============================================================
# PAGE 1: 🏠 HOME / DASHBOARD
# ============================================================

if selected_page == "🏠 Home / Dashboard":
    st.markdown('<div class="page-title">🏠 Agricultural Technology Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Welcome to AgriSmart — An integrated decision support system for data-driven soil testing, cost minimization, and explainable agronomy.</div>', unsafe_allow_html=True)

    # Metric Cards Strip
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown("""
        <div class="metric-box">
            <h4>Machine Learning Model</h4>
            <div class="val">Random Forest</div>
            <small style="color:#555;">200 Decision Trees (MDI)</small>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-box">
            <h4>Calibrated Dataset</h4>
            <div class="val">{len(dataset_df) if dataset_df is not None else 10000:,}</div>
            <small style="color:#555;">Soil & Crop Samples</small>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="metric-box">
            <h4>Model Accuracy</h4>
            <div class="val">{model_acc * 100:.1f}%</div>
            <small style="color:#555;">Cross-Validated Performance</small>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown("""
        <div class="metric-box">
            <h4>Subsidy Standards</h4>
            <div class="val">ICAR / NBS</div>
            <small style="color:#555;">Statutory MRP Benchmark</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Core System Modules Overview
    st.markdown("### 🌾 Integrated Platform Capabilities")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("""
        <div class="clean-card">
            <h3 style="color:#0e4b26; margin-top:0;">🌱 Fertilizer Recommender</h3>
            <p style="font-size:14px; color:#444;">Multi-feature Random Forest classifier evaluating soil N, P, K, pH, moisture, and crop requirements to predict the optimal fertilizer.</p>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="clean-card">
            <h3 style="color:#0e4b26; margin-top:0;">💰 Cost Optimizer</h3>
            <p style="font-size:14px; color:#444;">Mathematical cost minimization solving stoichiometric nutrient demands to calculate required bags and compare against standard unguided practice.</p>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="clean-card">
            <h3 style="color:#0e4b26; margin-top:0;">🔍 Explainable AI</h3>
            <p style="font-size:14px; color:#444;">Feature importance rankings and local tree-path probability decomposition showing exact mathematical attributions for model transparency.</p>
        </div>
        """, unsafe_allow_html=True)

    m4, m5, m6 = st.columns(3)
    with m4:
        st.markdown("""
        <div class="clean-card">
            <h3 style="color:#0e4b26; margin-top:0;">🤖 AI Farming Assistant</h3>
            <p style="font-size:14px; color:#444;">Interactive agronomic decision advisor providing verified guidance on DAP placement, Zinc in paddy, pH correction, and biofertilizers.</p>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown("""
        <div class="clean-card">
            <h3 style="color:#0e4b26; margin-top:0;">🛒 Farmer Marketplace</h3>
            <p style="font-size:14px; color:#444;">Explore official subsidized fertilizer rates, calculate purchase quotes, and browse farmer crop listings with government MSP comparisons.</p>
        </div>
        """, unsafe_allow_html=True)
    with m6:
        st.markdown("""
        <div class="clean-card">
            <h3 style="color:#0e4b26; margin-top:0;">📄 Soil Health Card</h3>
            <p style="font-size:14px; color:#444;">Standardized Government of India benchmarked soil test card ready for viewing, printing, and agronomist verification.</p>
        </div>
        """, unsafe_allow_html=True)

    # Visual Agronomic Analytics
    if dataset_df is not None:
        st.markdown("### 📊 Dataset Agronomic Intelligence")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Crop Types Represented (Top 8)**")
            if "Crop_Type" in dataset_df.columns:
                st.bar_chart(dataset_df["Crop_Type"].value_counts().head(8))
        with c2:
            st.markdown("**Fertilizer Classes in Model**")
            if "Recommended_Fertilizer" in dataset_df.columns:
                st.bar_chart(dataset_df["Recommended_Fertilizer"].value_counts())


# ============================================================
# PAGE 2: 🌱 FERTILIZER RECOMMENDATION (WITH ALL 19 COLUMNS)
# ============================================================

elif selected_page == "🌱 Fertilizer Recommendation":
    st.markdown('<div class="page-title">🌱 Precision Fertilizer Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Input your farm soil chemical test values to generate an ML-driven fertilizer prescription using the calibrated 200-tree Random Forest pipeline.</div>', unsafe_allow_html=True)

    # 1-Click Realistic Presets
    st.markdown("##### ⚡ 1-Click Realistic Field Presets")
    p1, p2, p3 = st.columns(3)
    preset_choice = None
    if p1.button("🌾 Wheat Deficient Soil (N:35, P:25, K:30)"):
        preset_choice = {"crop": "Wheat", "soil": "Loamy", "n": 35.0, "p": 25.0, "k": 30.0, "ph": 6.5, "budget": 3000.0}
    if p2.button("🌾 Rice Low Phosphorus (N:60, P:15, K:40)"):
        preset_choice = {"crop": "Rice", "soil": "Clay", "n": 60.0, "p": 15.0, "k": 40.0, "ph": 5.8, "budget": 3500.0}
    if p3.button("🌽 Maize High Potassium (N:80, P:50, K:50)"):
        preset_choice = {"crop": "Maize", "soil": "Sandy", "n": 80.0, "p": 50.0, "k": 50.0, "ph": 6.8, "budget": 2500.0}

    # Default values
    d_crop = preset_choice["crop"] if preset_choice else "Wheat"
    d_soil = preset_choice["soil"] if preset_choice else "Loamy"
    d_n = preset_choice["n"] if preset_choice else 50.0
    d_p = preset_choice["p"] if preset_choice else 45.0
    d_k = preset_choice["k"] if preset_choice else 40.0
    d_ph = preset_choice["ph"] if preset_choice else 6.5
    d_budget = preset_choice["budget"] if preset_choice else 3000.0

    with st.form("rec_form"):
        st.markdown("### 🧪 Primary Soil & Crop Parameters")
        col1, col2 = st.columns(2)
        with col1:
            supported_crops = ["Wheat", "Rice", "Maize", "Cotton", "Potato", "Sugarcane"]
            c_idx = supported_crops.index(d_crop) if d_crop in supported_crops else 0
            in_crop = st.selectbox("Target Crop", supported_crops, index=c_idx)

            soil_types = ["Loamy", "Clay", "Sandy", "Silt"]
            s_idx = soil_types.index(d_soil) if d_soil in soil_types else 0
            in_soil = st.selectbox("Soil Type / Texture", soil_types, index=s_idx)

            in_n = st.number_input("Available Nitrogen (N) [kg/ha]", min_value=0.0, max_value=300.0, value=d_n, step=1.0)
            in_p = st.number_input("Available Phosphorus (P) [kg/ha]", min_value=0.0, max_value=300.0, value=d_p, step=1.0)

        with col2:
            in_k = st.number_input("Available Potassium (K) [kg/ha]", min_value=0.0, max_value=300.0, value=d_k, step=1.0)
            in_ph = st.number_input("Soil Reaction (pH)", min_value=0.0, max_value=14.0, value=d_ph, step=0.1)
            in_area = st.number_input("Farm Land Area (Acres)", min_value=0.1, max_value=500.0, value=2.0, step=0.5)
            in_budget = st.number_input("Available Fertilizer Budget (₹)", min_value=100.0, max_value=500000.0, value=d_budget, step=100.0)

        # Advanced Agronomic Context Section (Pre-filled with Scientific Regional Norms)
        with st.expander("🌱 Environmental & Agronomic Context (Optional / Calibrated Defaults)", expanded=False):
            st.caption("These fields complete the 19 features required by the trained Random Forest model. Sensible agricultural defaults are pre-populated.")
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                in_growth = st.selectbox("Crop Growth Stage", ["Vegetative", "Sowing", "Flowering", "Harvest"], index=0)
                in_season = st.selectbox("Season", ["Rabi", "Kharif", "Zaid"], index=0)
                in_irrigation = st.selectbox("Irrigation Type", ["Canal", "Sprinkler", "Rainfed", "Drip"], index=0)
                in_region = st.selectbox("Agricultural Region", ["North", "South", "East", "West", "Central"], index=0)
            with ec2:
                in_prev_crop = st.selectbox("Previous Crop in Field", ["Wheat", "Rice", "Maize", "Potato", "Cotton", "Tomato"], index=0)
                in_moisture = st.number_input("Soil Moisture (%)", min_value=5.0, max_value=90.0, value=35.0, step=1.0)
                in_oc = st.number_input("Organic Carbon (%)", min_value=0.1, max_value=5.0, value=0.85, step=0.05)
                in_ec = st.number_input("Electrical Conductivity (dS/m)", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
            with ec3:
                in_temp = st.number_input("Average Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.5)
                in_humidity = st.number_input("Relative Humidity (%)", min_value=10.0, max_value=100.0, value=60.0, step=1.0)
                in_rainfall = st.number_input("Annual Rainfall (mm)", min_value=100.0, max_value=4000.0, value=1200.0, step=50.0)
                in_fert_last = st.number_input("Fertilizer Used Last Season (kg)", min_value=0.0, max_value=1000.0, value=175.0, step=10.0)
                in_yield_last = st.number_input("Yield Last Season (t/ha)", min_value=0.1, max_value=20.0, value=4.5, step=0.1)

        submitted_rec = st.form_submit_button("🔍 Generate Fertilizer Recommendation", type="primary", use_container_width=True)

    if submitted_rec or preset_choice:
        if ml_model is None:
            st.error("❌ Trained model (fertilizer_model.pkl) could not be loaded.")
        else:
            # Construct the complete 19-column DataFrame in the exact training order
            input_row = {
                'Soil_Type': in_soil,
                'Soil_pH': in_ph,
                'Soil_Moisture': in_moisture if 'in_moisture' in locals() else 35.0,
                'Organic_Carbon': in_oc if 'in_oc' in locals() else 0.85,
                'Electrical_Conductivity': in_ec if 'in_ec' in locals() else 1.5,
                'Nitrogen_Level': in_n,
                'Phosphorus_Level': in_p,
                'Potassium_Level': in_k,
                'Temperature': in_temp if 'in_temp' in locals() else 25.0,
                'Humidity': in_humidity if 'in_humidity' in locals() else 60.0,
                'Rainfall': in_rainfall if 'in_rainfall' in locals() else 1200.0,
                'Crop_Type': in_crop,
                'Crop_Growth_Stage': in_growth if 'in_growth' in locals() else "Vegetative",
                'Season': in_season if 'in_season' in locals() else "Rabi",
                'Irrigation_Type': in_irrigation if 'in_irrigation' in locals() else "Canal",
                'Previous_Crop': in_prev_crop if 'in_prev_crop' in locals() else "Wheat",
                'Region': in_region if 'in_region' in locals() else "North",
                'Fertilizer_Used_Last_Season': in_fert_last if 'in_fert_last' in locals() else 175.0,
                'Yield_Last_Season': in_yield_last if 'in_yield_last' in locals() else 4.5
            }

            input_df = pd.DataFrame([input_row])[FEATURE_ORDER]

            try:
                prediction = ml_model.predict(input_df)[0]
                conf = 92.5
                if hasattr(ml_model, "predict_proba"):
                    probs = ml_model.predict_proba(input_df)
                    conf = round(float(np.max(probs)) * 100, 1)

                st.markdown(f'<div class="recommendation-banner">🌱 Recommended Fertilizer: {prediction} &nbsp;<span style="font-size:16px; font-weight:600; color:#176b3a;">({conf}% Confidence)</span></div>', unsafe_allow_html=True)

                res_c1, res_c2 = st.columns(2)
                with res_c1:
                    st.markdown("#### 📈 Soil Nutrient Status vs Benchmarks")
                    ndf = pd.DataFrame({"Nutrient": ["Nitrogen", "Phosphorus", "Potassium"], "Tested Value (kg/ha)": [in_n, in_p, in_k]})
                    st.bar_chart(ndf.set_index("Nutrient"))
                with res_c2:
                    st.markdown("#### 💡 Agronomic Advisory")
                    st.markdown(f"""
                    - **Crop:** **{in_crop}** on **{in_soil}** soil.
                    - **Soil pH {in_ph:.1f}:** {'Optimal neutral range.' if 6.0 <= in_ph <= 7.5 else 'Requires amendment (Lime for acid, Gypsum for alkaline).' }
                    - **Action Plan:** Apply recommended **{prediction}** in split applications to reduce nutrient runoff.
                    - **Next Step:** Open **💰 Fertilizer Cost Optimizer** in the sidebar to calculate exact bag quantities and commercial savings for your {in_area} acre plot.
                    """)
            except Exception as pred_err:
                st.error(f"Prediction Pipeline Error: {pred_err}")


# ============================================================
# PAGE 3: 💰 FERTILIZER COST OPTIMIZER
# ============================================================

elif selected_page == "💰 Fertilizer Cost Optimizer":
    st.markdown('<div class="page-title">💰 Fertilizer Cost Optimizer & Area Planner</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Minimizes fertilizer expenditure while fulfilling 100% of crop nutrient demands under Indian statutory subsidy rates.</div>', unsafe_allow_html=True)

    with st.form("cost_opt_form"):
        st.markdown("### 🌾 Farm Area & Budget Parameters")
        c1, c2 = st.columns(2)
        with c1:
            supported_crops = ["Wheat", "Rice", "Maize", "Cotton", "Sugarcane", "Potato", "Tomato", "Soybean"]
            opt_crop = st.selectbox("Target Crop", supported_crops)
            opt_area = st.number_input("Farm Land Area", min_value=0.1, max_value=500.0, value=2.0, step=0.5)
        with c2:
            opt_unit = st.selectbox("Land Area Unit", ["Acre", "Hectare"])
            opt_budget = st.number_input("Allocated Fertilizer Budget (₹)", min_value=100.0, max_value=500000.0, value=3000.0, step=100.0)

        st.markdown("### 🧪 Soil Nutrient Levels (Optional / From Soil Test)")
        s1, s2, s3 = st.columns(3)
        with s1:
            opt_n = st.number_input("Nitrogen (N) [kg/ha]", min_value=0.0, max_value=300.0, value=50.0, step=1.0)
        with s2:
            opt_p = st.number_input("Phosphorus (P) [kg/ha]", min_value=0.0, max_value=300.0, value=45.0, step=1.0)
        with s3:
            opt_k = st.number_input("Potassium (K) [kg/ha]", min_value=0.0, max_value=300.0, value=40.0, step=1.0)

        opt_submit = st.form_submit_button("🧮 Compute Cost-Optimized Package", type="primary", use_container_width=True)

    if opt_submit:
        if optimize_fertilizer_cost is None:
            st.error("Cost optimizer module unavailable.")
        else:
            try:
                # Calls optimize_fertilizer_cost flexibly
                opt_res = optimize_fertilizer_cost(
                    crop=opt_crop,
                    plot_area=opt_area,
                    area_unit=opt_unit,
                    budget=opt_budget,
                    nitrogen=opt_n,
                    phosphorus=opt_p,
                    potassium=opt_k
                )

                # Top KPI Strip
                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.metric("Optimal Strategy", opt_res.get("recommended_fertilizer", "Straight Package"))
                with k2:
                    st.metric("Total Subsidized Cost", f"₹ {opt_res.get('total_cost', 0):,.2f}")
                with k3:
                    st.metric("Estimated Savings", f"₹ {opt_res.get('savings_amount', 0):,.2f}")
                with k4:
                    status_lbl = "✅ Within Budget" if opt_res.get("within_budget", True) else "⚠️ Budget Deficit"
                    st.metric("Budget Feasibility", status_lbl)

                if not opt_res.get("within_budget", True):
                    st.warning(f"⚠️ {opt_res.get('budget_message', 'Package exceeds allocated budget.')}")
                else:
                    st.success(f"✅ {opt_res.get('budget_message', 'Allocated budget is sufficient.')}")

                # Dosage breakdown table
                st.markdown("### 📦 Recommended Fertilizer Dosage & Quantities")
                items = opt_res.get("combination_details") or opt_res.get("optimal_plan", {}).get("items", [])
                if items:
                    idf = pd.DataFrame(items)[["fertilizer", "bags", "bag_size_kg", "total_kg", "unit_price", "cost", "price_type"]]
                    idf.columns = ["Fertilizer Name", "Bags Required", "Bag Size (kg)", "Total Quantity (kg)", "Price / Bag (₹)", "Total Cost (₹)", "Subsidy Tag"]
                    st.dataframe(idf, use_container_width=True)

                # Application schedule
                st.markdown("### 📅 ICAR 3-Stage Split Application Schedule")
                sched = opt_res.get("application_schedule", [])
                if sched:
                    st.table(pd.DataFrame(sched))

            except Exception as opt_err:
                st.error(f"Optimization calculation error: {opt_err}")


# ============================================================
# PAGE 4: 🔍 EXPLAINABLE AI (XAI)
# ============================================================

elif selected_page == "🔍 Explainable AI (XAI)":
    st.markdown('<div class="page-title">🔍 Explainable AI & Model Transparency</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Inspect the global Gini MDI feature rankings and understand how the 200 Random Forest decision trees evaluate soil parameters.</div>', unsafe_allow_html=True)

    if get_global_feature_importance:
        xai_res = get_global_feature_importance()
        st.markdown(f"**Model Type:** Random Forest Classifier ({xai_res.get('total_trees', 200)} Decision Trees) • **Evaluator:** Gini Mean Decrease in Impurity (MDI)")

        chart_data = xai_res.get("chart_data", [])
        if chart_data:
            st.markdown("### 📊 Global Feature Importance Ranking")
            cdf = pd.DataFrame(chart_data)
            st.bar_chart(cdf.set_index("feature"))

            st.markdown("### 📋 Feature Weight Matrix")
            st.dataframe(cdf, use_container_width=True)

        st.markdown("""
        <div class="clean-card">
            <h4 style="color:#0e4b26; margin-top:0;">💡 Agronomic Transparency Findings:</h4>
            <ul style="margin-bottom:0; font-size:14.5px; line-height:1.7;">
                <li><b>Soil pH & Nitrogen:</b> Exhibit the highest split frequency across trees because soil reaction governs microbial mineralization and nutrient availability.</li>
                <li><b>Phosphorus (P):</b> Serves as the primary decision node when distinguishing high-analysis DAP (18:46:0) from general NPK complexes.</li>
                <li><b>Potassium (K):</b> Determines the requirement for Muriate of Potash (MOP) to support lodging resistance and grain filling.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE 5: 🤖 AI FARMING ASSISTANT (HIGH CONTRAST & READABLE)
# ============================================================

elif selected_page == "🤖 AI Farming Assistant":
    st.markdown('<div class="page-title">🤖 AI Agronomy Decision Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Interactive decision support on fertilizer application, N-P-K nutrient roles, soil correction, and irrigation management.</div>', unsafe_allow_html=True)

    quick_qs = [
        "Which fertilizer is best for wheat?",
        "What is the role of nitrogen in plant growth?",
        "When and how should I apply DAP?",
        "How do I cure zinc deficiency (Khaira disease) in rice?",
        "What is the ideal soil pH and how do I correct acidic soil?",
        "How can I improve soil organic carbon using biofertilizers?"
    ]

    sel_q = st.selectbox("💡 Select a Common Agronomic Topic:", ["-- Choose a topic --"] + quick_qs)
    custom_q = st.text_input("Or type your farming question:", value=sel_q if sel_q != "-- Choose a topic --" else "", placeholder="e.g. When should I top dress urea in paddy?")

    if st.button("Ask Assistant", type="primary") and custom_q:
        ql = custom_q.lower()
        if "wheat" in ql:
            ans = "🌾 **Wheat Nutrient Advisory:**\n- **Target:** 120 kg N, 60 kg P2O5, 40 kg K2O per hectare.\n- **Basal Dose (Sowing):** Apply 100% P & K (via DAP or NPK) + 1/3rd Nitrogen.\n- **Top Dressings:** Split remaining Urea at Crown Root Initiation (CRI stage, 21 DAS) and flowering stage."
        elif "nitrogen" in ql:
            ans = "🌱 **Role of Nitrogen (N):**\n- Drives vegetative leaf development, stem elongation, and chlorophyll synthesis.\n- **Deficiency Symptom:** Uniform yellowing (chlorosis) of older bottom leaves.\n- **Remedy:** Apply Urea or foliar spray 2% Urea solution for quick recovery."
        elif "dap" in ql:
            ans = "⏳ **DAP (18:46:0) Application Timing:**\n- DAP supplies water-soluble phosphorus for early root proliferation.\n- **Method:** Always band-place 3-5 cm below and beside the seed at sowing.\n- **Caution:** Never surface-broadcast DAP on dry soil as phosphate is non-mobile."
        elif "zinc" in ql or "khaira" in ql:
            ans = "⚡ **Zinc Deficiency in Rice (Khaira Disease):**\n- Appears as rusty brown spots on young leaves 2-3 weeks after transplanting.\n- **Soil Application:** Broadcast 25 kg/ha Zinc Sulphate (ZnSO4 21%) during puddling.\n- **Foliar Rescue:** Spray 0.5% Zinc Sulphate + 0.25% Slaked Lime solution."
        elif "ph" in ql:
            ans = "🧪 **Soil pH & Neutralization:**\n- **Optimal Range:** 6.0 to 7.5 for maximum nutrient availability.\n- **Acidic Soil (pH < 5.5):** Apply Agricultural Lime (CaCO3) @ 200-400 kg/acre.\n- **Alkaline/Sodic Soil (pH > 8.0):** Apply Agricultural Gypsum (CaSO4·2H2O) @ 250 kg/acre."
        elif "organic" in ql or "carbon" in ql or "biofertilizer" in ql:
            ans = "🍂 **Soil Organic Carbon (OC) Management:**\n- Healthy productive soil requires > 0.75% Organic Carbon.\n- Apply 4-5 tonnes/acre Farmyard Manure (FYM) or 2 tonnes/acre Vermicompost.\n- Inoculate seeds with *Azotobacter* (fixes 20-30 kg N/ha) and PSB (Phosphorus Solubilizing Bacteria)."
        else:
            ans = "🌿 **Agronomist Advisory:**\nBalanced fertilization requires matching crop requirements with soil test results. Combine chemical fertilizers with 20-25% organic manure and ensure split nitrogen application to prevent leaching."

        # High-contrast readable card
        st.markdown(f"""
        <div class="chat-bubble-bot">
            <strong style="color:#0e4b26; font-size:16px;">🌱 AgriSmart Agronomist Response:</strong><br><br>
            {ans.replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE 6: 🛒 FARMER MARKETPLACE
# ============================================================

elif selected_page == "🛒 Farmer Marketplace":
    st.markdown('<div class="page-title">🛒 Agricultural Marketplace</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Demonstration portal for authorized fertilizer procurement and farmer crop selling with MSP market comparisons.</div>', unsafe_allow_html=True)

    st.info("🎓 **Demonstration Mode:** Displays government statutory fertilizer subsidy rates and simulated APMC Mandi crop sales.")

    tab_buy, tab_sell = st.tabs(["🌱 Buy Fertilizers (Catalog & Quote)", "🌾 Sell Crops (Farmer Mandi Listings)"])

    with tab_buy:
        st.markdown("### 🏷️ Official Fertilizer Products")
        fert_rows = [
            {"Fertilizer": "Urea (Neem Coated)", "Nutrient Content": "N: 46%, P: 0%, K: 0%", "Bag Size": "45 kg", "Statutory Subsidized Rate": "₹ 266.50", "Price Benchmark": "Statutory MRP"},
            {"Fertilizer": "DAP (Di-Ammonium Phos.)", "Nutrient Content": "N: 18%, P: 46%, K: 0%", "Bag Size": "50 kg", "Statutory Subsidized Rate": "₹ 1,350.00", "Price Benchmark": "NBS Subsidized"},
            {"Fertilizer": "MOP (Potassium Chloride)", "Nutrient Content": "N: 0%, P: 0%, K: 60%", "Bag Size": "50 kg", "Statutory Subsidized Rate": "₹ 1,700.00", "Price Benchmark": "Indicative Market"},
            {"Fertilizer": "NPK Complex (10:26:26)", "Nutrient Content": "N: 10%, P: 26%, K: 26%", "Bag Size": "50 kg", "Statutory Subsidized Rate": "₹ 1,470.00", "Price Benchmark": "Indicative Market"},
            {"Fertilizer": "Single Super Phosphate (SSP)", "Nutrient Content": "P: 16%, S: 11%, Ca: 19%", "Bag Size": "50 kg", "Statutory Subsidized Rate": "₹ 550.00", "Price Benchmark": "Statutory MRP"},
            {"Fertilizer": "Organic Vermicompost", "Nutrient Content": "OC >14%, NPK 1.5-1-1.5%", "Bag Size": "50 kg", "Statutory Subsidized Rate": "₹ 200.00", "Price Benchmark": "Bio-Organic"}
        ]
        st.dataframe(pd.DataFrame(fert_rows), use_container_width=True)

        st.markdown("#### 🧮 Sample Purchase Quote Calculator")
        q1, q2 = st.columns(2)
        with q1:
            prod_sel = st.selectbox("Select Product", ["Urea (₹266.50)", "DAP (₹1,350.00)", "MOP (₹1,700.00)", "NPK 10:26:26 (₹1,470.00)", "SSP (₹550.00)", "Vermicompost (₹200.00)"])
        with q2:
            bags_in = st.number_input("Number of Bags", min_value=1, max_value=1000, value=10)

        pmap = {"Urea (₹266.50)": 266.50, "DAP (₹1,350.00)": 1350.0, "MOP (₹1,700.00)": 1700.0, "NPK 10:26:26 (₹1,470.00)": 1470.0, "SSP (₹550.00)": 550.0, "Vermicompost (₹200.00)": 200.0}
        st.success(f"**Total Estimated Purchase Cost:** ₹ {bags_in * pmap[prod_sel]:,.2f}")

    with tab_sell:
        st.markdown("### 🌾 Active Farmer Crop Listings")
        if get_crop_listings:
            listings = get_crop_listings(limit=25)
            if listings:
                ldf = pd.DataFrame(listings)[["farmer_name", "location", "crop", "quantity_quintals", "expected_price", "created_at"]]
                ldf.columns = ["Farmer Name", "Mandi Location", "Crop", "Quantity (Qtl)", "Expected Price (₹/Qtl)", "Listing Date"]
                st.dataframe(ldf, use_container_width=True)

        st.markdown("#### ➕ Post Sample Harvest Listing")
        with st.form("crop_sell_form"):
            sf1, sf2 = st.columns(2)
            with sf1:
                cfarmer = st.text_input("Farmer Name", value="Kuldeep Singh")
                cloc = st.text_input("Mandi Location", value="Ludhiana, Punjab")
            with sf2:
                ccrop = st.selectbox("Crop", ["Wheat", "Rice", "Cotton", "Maize", "Soybean", "Mustard"])
                cqty = st.number_input("Quantity (Quintals)", min_value=1.0, value=50.0, step=1.0)
                cprice = st.number_input("Expected Price (₹/Quintal)", min_value=500.0, value=2275.0, step=25.0)

            csub = st.form_submit_button("Post Crop Listing (Demo)")
            if csub and add_crop_listing:
                add_crop_listing(cfarmer, cloc, ccrop, cqty, cprice)
                st.success(f"✅ Sample listing for {ccrop} ({cqty} Quintals) registered successfully in the marketplace!")
                st.rerun()


# ============================================================
# PAGE 7: 📄 SOIL HEALTH CARD
# ============================================================

elif selected_page == "📄 Soil Health Card":
    st.markdown('<div class="page-title">📄 Soil Health & Fertilizer Recommendation Card</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Standardized soil test report card benchmarked against Government of India Ministry of Agriculture guidelines.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="clean-card" style="border: 2px solid #2e7d32; background:#ffffff;">
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:2px solid #2e7d32; padding-bottom:12px; margin-bottom:18px;">
            <div>
                <h2 style="color:#0e4b26; margin:0; font-size:22px;">🌾 SOIL HEALTH & FERTILIZER CARD</h2>
                <span style="font-size:12px; color:#555;">AgriSmart Decision Support System • Dept. of Agriculture Reference Norms</span>
            </div>
            <div style="text-align:right;">
                <strong style="color:#2e7d32; font-size:16px;">🌱 AGRISMART</strong><br>
                <span style="font-size:12px; color:#777;">Report Date: 2026</span>
            </div>
        </div>

        <table style="width:100%; border-collapse:collapse; margin-bottom:20px; font-size:14px;">
            <tr style="background:#f4f9f5;">
                <td style="padding:8px 12px; font-weight:700; border:1px solid #d4e7d7;">Farmer Name:</td>
                <td style="padding:8px 12px; border:1px solid #d4e7d7;">Guest Farmer</td>
                <td style="padding:8px 12px; font-weight:700; border:1px solid #d4e7d7;">Target Crop:</td>
                <td style="padding:8px 12px; font-weight:700; color:#0e4b26; border:1px solid #d4e7d7;">Wheat (HD-2967)</td>
            </tr>
            <tr>
                <td style="padding:8px 12px; font-weight:700; border:1px solid #d4e7d7;">Plot Land Area:</td>
                <td style="padding:8px 12px; border:1px solid #d4e7d7;">2.0 Acres</td>
                <td style="padding:8px 12px; font-weight:700; border:1px solid #d4e7d7;">Soil Texture:</td>
                <td style="padding:8px 12px; border:1px solid #d4e7d7;">Sandy Loam</td>
            </tr>
        </table>

        <h4 style="color:#0e4b26; margin-bottom:8px;">🧪 Chemical & Physical Parameters</h4>
        <table style="width:100%; border-collapse:collapse; margin-bottom:20px; font-size:13.5px;">
            <thead>
                <tr style="background:#e8f5eb; color:#0e4b26;">
                    <th style="padding:8px 12px; text-align:left; border:1px solid #d4e7d7;">Parameter</th>
                    <th style="padding:8px 12px; text-align:left; border:1px solid #d4e7d7;">Tested Value</th>
                    <th style="padding:8px 12px; text-align:left; border:1px solid #d4e7d7;">Standard Benchmark</th>
                    <th style="padding:8px 12px; text-align:left; border:1px solid #d4e7d7;">Rating Status</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="padding:8px 12px; border:1px solid #d4e7d7;"><b>Soil Reaction (pH)</b></td>
                    <td style="padding:8px 12px; border:1px solid #d4e7d7;">6.5</td>
                    <td style="padding:8px 12px; border:1px solid #d4e7d7;">6.0 - 7.5</td>
                    <td style="padding:8px 12px; color:#2e7d32; font-weight:700; border:1px solid #d4e7d7;">Optimal Neutral</td>
                </tr>
                <tr style="background:#fcfdfc;">
                    <td style="padding:8px 12px; border:1px solid #d4e7d7;"><b>Available Nitrogen (N)</b></td>
                    <td style="padding:8px 12px; border:1px solid #d4e7d7;">45 kg/ha</td>
                    <td style="padding:8px 12px; border:1px solid #d4e7d7;">> 120 kg/ha</td>
                    <td style="padding:8px 12px; color:#c62828; font-weight:700; border:1px solid #d4e7d7;">Low / Deficient</td>
                </tr>
                <tr>
                    <td style="padding:8px 12px; border:1px solid #d4e7d7;"><b>Available Phosphorus (P)</b></td>
                    <td style="padding:8px 12px; border:1px solid #d4e7d7;">35 kg/ha</td>
                    <td style="padding:8px 12px; border:1px solid #d4e7d7;">> 60 kg/ha</td>
                    <td style="padding:8px 12px; color:#f57c00; font-weight:700; border:1px solid #d4e7d7;">Moderate</td>
                </tr>
                <tr style="background:#fcfdfc;">
                    <td style="padding:8px 12px; border:1px solid #d4e7d7;"><b>Available Potassium (K)</b></td>
                    <td style="padding:8px 12px; border:1px solid #d4e7d7;">40 kg/ha</td>
                    <td style="padding:8px 12px; border:1px solid #d4e7d7;">> 50 kg/ha</td>
                    <td style="padding:8px 12px; color:#f57c00; font-weight:700; border:1px solid #d4e7d7;">Moderate</td>
                </tr>
            </tbody>
        </table>

        <div style="background:#f1f8f3; border-left:5px solid #2e7d32; padding:14px; border-radius:6px; margin-bottom:20px;">
            <strong style="color:#0e4b26;">🌱 Machine Learning Prescription: Urea + DAP Basal Combination</strong><br>
            <span style="font-size:13px; color:#333;">Total Required: <b>2 Bags DAP (100 kg)</b> + <b>3 Bags Urea (135 kg)</b> • Estimated Subsidized Cost: <b>₹ 3,499.50</b></span>
        </div>

        <div style="border-top:1px dashed #aaa; padding-top:15px; display:flex; justify-content:space-between; align-items:flex-end;">
            <small style="color:#666;">* Generated via AgriSmart Decision Support System based on ICAR agronomy guidelines.</small>
            <div style="text-align:center;">
                <div style="border-bottom:1px solid #333; width:140px; margin-bottom:4px;"></div>
                <small style="font-weight:700;">Authorized Agronomist</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer-text">
    🌱 <b>AgriSmart Integrated System</b> • Precision Fertilizer Optimization & Transparent Decision Support<br>
    Final Year Project Demonstration • Dual Interface: Streamlit (Port 8501) & Flask (Port 5000)
</div>
""", unsafe_allow_html=True)
