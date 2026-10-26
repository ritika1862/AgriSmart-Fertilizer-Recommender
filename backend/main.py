from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import traceback
import math
import sys
from pathlib import Path
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Import database module
from database.database import (
    create_user,
    authenticate_user,
    save_recommendation_record,
    get_recommendations,
    delete_recommendation,
    get_aggregated_stats,
    add_crop_listing,
    get_crop_listings
)

# Import cost optimizer engine
try:
    from backend.optimizer import optimize_fertilizer_cost, get_fertilizer_catalog_list
except ImportError:
    from optimizer import optimize_fertilizer_cost, get_fertilizer_catalog_list

# Import explainable AI engine
try:
    from backend.explainability import get_global_feature_importance, explain_individual_prediction
except ImportError:
    from explainability import get_global_feature_importance, explain_individual_prediction



# ============================================================
# FLASK APPLICATION SETUP
# ============================================================

app = Flask(__name__, static_folder="../frontend", static_url_path="")
app.secret_key = "fertilizer_optimizer_secure_session_key_2026"
CORS(app, supports_credentials=True)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "fertilizer_model.pkl"

# ============================================================
# FERTILIZER & AGRONOMY MASTER KNOWLEDGE
# ============================================================

FERTILIZER_MASTER = {
    "Urea": {
        "n_percent": 46.0, "p_percent": 0.0, "k_percent": 0.0,
        "bag_size_kg": 45, "price_per_bag": 266.50,
        "description": "High-nitrogen quick-release fertilizer essential for vegetative growth and green foliage.",
        "best_for": "Nitrogen deficiency, leafy growth, tillering stage.",
        "application_method": "Top dressing / Split broadcasting before irrigation."
    },
    "DAP": {
        "n_percent": 18.0, "p_percent": 46.0, "k_percent": 0.0,
        "bag_size_kg": 50, "price_per_bag": 1350.00,
        "description": "Di-Ammonium Phosphate provides concentrated phosphorus and starter nitrogen for strong root establishment.",
        "best_for": "Phosphorus deficiency, root growth, basal application at sowing.",
        "application_method": "Basal dose placed 3-5 cm below and beside the seed."
    },
    "MOP": {
        "n_percent": 0.0, "p_percent": 0.0, "k_percent": 60.0,
        "bag_size_kg": 50, "price_per_bag": 1700.00,
        "description": "Muriate of Potash (Potassium Chloride) enhances disease resistance, drought tolerance, and grain/fruit filling.",
        "best_for": "Potassium deficiency, fruit quality, lodging resistance.",
        "application_method": "Basal dose during land preparation or split dose at flowering."
    },
    "NPK": {
        "n_percent": 10.0, "p_percent": 26.0, "k_percent": 26.0,
        "bag_size_kg": 50, "price_per_bag": 1470.00,
        "description": "Complex fertilizer supplying balanced macro-nutrients (N-P-K) in equal/optimized proportions.",
        "best_for": "Balanced nutrition across all growth stages.",
        "application_method": "Basal application and secondary top dressing."
    },
    "SSP": {
        "n_percent": 0.0, "p_percent": 16.0, "k_percent": 0.0,
        "bag_size_kg": 50, "price_per_bag": 550.00,
        "description": "Single Super Phosphate supplies water-soluble phosphorus along with 11% Sulphur and 19% Calcium.",
        "best_for": "Oilseeds, pulses, sulphur-deficient soil, low budget basal application.",
        "application_method": "Basal soil application during last ploughing."
    },
    "Zinc Sulphate": {
        "n_percent": 0.0, "p_percent": 0.0, "k_percent": 0.0,
        "bag_size_kg": 25, "price_per_bag": 650.00,
        "description": "Essential micronutrient fertilizer to treat zinc deficiency (Khaira disease in paddy, white bud in maize).",
        "best_for": "Zinc deficient soils, alkaline soils, cereals.",
        "application_method": "Soil application at sowing or foliar spray at 0.5% concentration."
    },
    "Compost": {
        "n_percent": 1.5, "p_percent": 1.0, "k_percent": 1.5,
        "bag_size_kg": 50, "price_per_bag": 200.00,
        "description": "Organic farmyard manure / vermicompost improving soil organic carbon, moisture retention, and microbial health.",
        "best_for": "Low organic carbon soils, degraded soils, all crops.",
        "application_method": "Incorporate thoroughly into top 15cm soil 2-3 weeks before sowing."
    }
}

CROP_STANDARDS = {
    "Wheat": {"target_n": 120, "target_p": 60, "target_k": 40, "ideal_ph_min": 6.0, "ideal_ph_max": 7.5},
    "Rice": {"target_n": 100, "target_p": 50, "target_k": 50, "ideal_ph_min": 5.5, "ideal_ph_max": 7.0},
    "Maize": {"target_n": 120, "target_p": 60, "target_k": 50, "ideal_ph_min": 5.8, "ideal_ph_max": 7.2},
    "Cotton": {"target_n": 100, "target_p": 50, "target_k": 50, "ideal_ph_min": 6.0, "ideal_ph_max": 8.0},
    "Sugarcane": {"target_n": 150, "target_p": 75, "target_k": 75, "ideal_ph_min": 6.0, "ideal_ph_max": 7.5},
    "Potato": {"target_n": 120, "target_p": 80, "target_k": 100, "ideal_ph_min": 5.0, "ideal_ph_max": 6.5},
    "Tomato": {"target_n": 100, "target_p": 60, "target_k": 60, "ideal_ph_min": 6.0, "ideal_ph_max": 7.0},
    "Soybean": {"target_n": 30, "target_p": 60, "target_k": 40, "ideal_ph_min": 6.0, "ideal_ph_max": 7.0},
    "Groundnut": {"target_n": 25, "target_p": 50, "target_k": 40, "ideal_ph_min": 5.8, "ideal_ph_max": 6.8},
    "Canal": {"target_n": 80, "target_p": 40, "target_k": 40, "ideal_ph_min": 6.0, "ideal_ph_max": 7.5},
    "Other": {"target_n": 90, "target_p": 50, "target_k": 50, "ideal_ph_min": 6.0, "ideal_ph_max": 7.5}
}

# ============================================================
# LOAD MACHINE LEARNING MODEL & DATASET
# ============================================================

model = None
dataset = None

try:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print("ML Model loaded successfully from fertilizer_model.pkl")
except Exception as e:
    print(f"Warning: Could not load ML model: {e}")

try:
    csv_path = DATA_DIR / "fertilizer_recommendation.csv"
    if csv_path.exists():
        dataset = pd.read_csv(csv_path)
        print(f"Dataset loaded: {len(dataset)} records, {len(dataset.columns)} columns.")
except Exception as e:
    print(f"Warning: Could not load dataset CSV: {e}")

# ============================================================
# HELPER PARSERS & AGRONOMIC CALCULATORS
# ============================================================

def parse_val(data, *keys, default=0.0):
    for key in keys:
        if key in data and data[key] not in (None, "", "null"):
            try:
                return float(data[key])
            except (ValueError, TypeError):
                return str(data[key])
    return default

def calculate_agronomy_and_economics(input_data, primary_fertilizer):
    """
    Computes precise fertilizer dosage, nutrient gaps, economics,
    application schedule, and soil health status based on agronomic science.
    """
    crop = input_data.get("crop", "Wheat")
    nitrogen = float(input_data.get("nitrogen", 50.0))
    phosphorus = float(input_data.get("phosphorus", 50.0))
    potassium = float(input_data.get("potassium", 50.0))
    ph = float(input_data.get("soil_ph", 6.5))
    oc = float(input_data.get("organic_carbon", 0.5))
    moisture = float(input_data.get("moisture", 30.0))
    plot_area = max(0.1, float(input_data.get("plot_area", 1.0)))
    area_unit = input_data.get("area_unit", "Acre")
    growth_stage = input_data.get("growth_stage", "Vegetative")

    # Standardize area to acres
    area_acres = plot_area if area_unit.lower() == "acre" else plot_area * 2.47105

    crop_info = CROP_STANDARDS.get(crop, CROP_STANDARDS["Other"])
    req_n = crop_info["target_n"] * area_acres
    req_p = crop_info["target_p"] * area_acres
    req_k = crop_info["target_k"] * area_acres

    # Available nutrients (adjusted soil index)
    avail_n = nitrogen * area_acres
    avail_p = phosphorus * area_acres
    avail_k = potassium * area_acres

    # Deficits
    gap_n = max(0.0, req_n - avail_n)
    gap_p = max(0.0, req_p - avail_p)
    gap_k = max(0.0, req_k - avail_k)

    # Normalize primary fertilizer
    fert_key = primary_fertilizer.strip()
    if fert_key not in FERTILIZER_MASTER:
        for k in FERTILIZER_MASTER:
            if k.lower() in fert_key.lower():
                fert_key = k
                break
        else:
            fert_key = "Urea"

    fert_meta = FERTILIZER_MASTER[fert_key]
    bag_size = fert_meta["bag_size_kg"]
    price_per_bag = fert_meta["price_per_bag"]

    # Calculate dosage in kg based on primary nutrient supplied
    if fert_key == "Urea":
        dosage_kg = (gap_n / 0.46) if gap_n > 0 else (45 * area_acres)
    elif fert_key == "DAP":
        dosage_kg = (gap_p / 0.46) if gap_p > 0 else (50 * area_acres)
    elif fert_key == "MOP":
        dosage_kg = (gap_k / 0.60) if gap_k > 0 else (30 * area_acres)
    elif fert_key == "NPK":
        dosage_kg = (max(gap_n, gap_p, gap_k) / 0.26) if max(gap_n, gap_p, gap_k) > 0 else (50 * area_acres)
    elif fert_key == "SSP":
        dosage_kg = (gap_p / 0.16) if gap_p > 0 else (100 * area_acres)
    elif fert_key == "Zinc Sulphate":
        dosage_kg = 15.0 * area_acres
    elif fert_key == "Compost":
        dosage_kg = 2000.0 * area_acres
    else:
        dosage_kg = 50.0 * area_acres

    dosage_kg = round(max(10.0, dosage_kg), 1)
    bags_count = math.ceil(dosage_kg / bag_size) if bag_size > 0 else 1
    estimated_cost = round(bags_count * price_per_bag, 2)

    # Standard unguided application benchmark (approx ₹1800/acre blanket application)
    standard_cost = round(max(estimated_cost * 1.35, 1850.0 * area_acres), 2)
    savings_amount = round(max(150.0 * area_acres, standard_cost - estimated_cost), 2)

    # Soil Health Status Evaluation
    soil_alerts = []
    if ph < 5.5:
        soil_alerts.append("Strongly Acidic (Apply Agricultural Lime / Dolomite @ 250 kg/acre).")
    elif ph > 8.2:
        soil_alerts.append("Alkaline / Calcareous (Apply Agricultural Gypsum @ 200 kg/acre).")
    else:
        soil_alerts.append("Optimal pH range for nutrient assimilation.")

    if oc < 0.5:
        soil_alerts.append("Low Organic Carbon: Incorporate green manure or FYM/vermicompost.")
    elif oc >= 0.75:
        soil_alerts.append("High Organic Carbon: Excellent soil microbial activity.")

    if nitrogen < 35:
        soil_alerts.append("Severe Nitrogen Deficit: Leaf yellowing risk (Chlorosis).")
    if phosphorus < 25:
        soil_alerts.append("Low Phosphorus: Restricted root branching expected.")
    if potassium < 30:
        soil_alerts.append("Low Potassium: Weak stem lodging & drought susceptibility.")

    # Multi-stage Application Schedule
    schedule = [
        {
            "stage": "Stage 1: Basal Application (At Sowing / Transplanting)",
            "portion": "50% of Total Dose",
            "quantity_kg": round(dosage_kg * 0.5, 1),
            "instructions": f"Broadcast or band place {round(dosage_kg * 0.5, 1)} kg of {fert_key} along with full dose of Organic Compost during final land preparation."
        },
        {
            "stage": "Stage 2: First Top Dressing (Vegetative / Tillering Stage - 25-30 DAS)",
            "portion": "25% of Total Dose",
            "quantity_kg": round(dosage_kg * 0.25, 1),
            "instructions": f"Apply {round(dosage_kg * 0.25, 1)} kg of {fert_key} when soil has sufficient moisture. Follow with light irrigation."
        },
        {
            "stage": "Stage 3: Second Top Dressing (Panicle Initiation / Pre-Flowering - 50-60 DAS)",
            "portion": "25% of Total Dose",
            "quantity_kg": round(dosage_kg * 0.25, 1),
            "instructions": f"Apply remaining {round(dosage_kg * 0.25, 1)} kg of {fert_key} to boost grain filling and yield quality."
        }
    ]

    return {
        "dosage_kg": dosage_kg,
        "bags_count": bags_count,
        "bag_size_kg": bag_size,
        "price_per_bag": price_per_bag,
        "estimated_cost": estimated_cost,
        "standard_cost": standard_cost,
        "savings_amount": savings_amount,
        "crop_requirements": {"N": round(req_n, 1), "P": round(req_p, 1), "K": round(req_k, 1)},
        "soil_available": {"N": round(avail_n, 1), "P": round(avail_p, 1), "K": round(avail_k, 1)},
        "nutrient_gaps": {"N": round(gap_n, 1), "P": round(gap_p, 1), "K": round(gap_k, 1)},
        "soil_alerts": soil_alerts,
        "schedule": schedule,
        "description": fert_meta["description"],
        "best_for": fert_meta["best_for"],
        "application_method": fert_meta["application_method"],
        "fertilizer_npk": f"{int(fert_meta['n_percent'])}:{int(fert_meta['p_percent'])}:{int(fert_meta['k_percent'])}"
    }

# ============================================================
# PAGE ROUTING
# ============================================================

@app.route("/")
@app.route("/welcome")
def welcome_page():
    return send_from_directory(FRONTEND_DIR, "welcome.html")


@app.route("/home")
@app.route("/app")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/entry")
def entry_page():
    return send_from_directory(FRONTEND_DIR, "entry.html")
    
@app.route("/login")
def login_page():
    return send_from_directory(FRONTEND_DIR, "login.html")

@app.route("/signup")
def signup_page():
    return send_from_directory(FRONTEND_DIR, "signup.html")

   

@app.route("/recommend")
@app.route("/recommender")
def recommend_page():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/dashboard")
def dashboard_page():
    return send_from_directory(FRONTEND_DIR, "dashboard.html")

@app.route("/ai-assistant")
@app.route("/ai_assistant")
def ai_assistant_page():
    return send_from_directory(FRONTEND_DIR, "ai_assistant.html")

@app.route("/soil-card")
@app.route("/soil_card")
def soil_card_page():
    return send_from_directory(FRONTEND_DIR, "soil_card.html")

@app.route("/optimizer")
@app.route("/cost-optimizer")
@app.route("/cost_optimizer")
def cost_optimizer_page():
    return send_from_directory(FRONTEND_DIR, "cost_optimizer.html")

@app.route("/explainability")
@app.route("/xai")
def explainability_page():
    return send_from_directory(FRONTEND_DIR, "explainability.html")

@app.route("/marketplace")
@app.route("/market")
def marketplace_page():
    return send_from_directory(FRONTEND_DIR, "marketplace.html")

# ============================================================
# MARKETPLACE APIS (DEMONSTRATION CATALOG & LISTINGS)
# ============================================================

@app.route("/api/marketplace/fertilizers")
def api_marketplace_fertilizers():
    """Returns official fertilizer products list with specifications and prices."""
    try:
        catalog = get_fertilizer_catalog_list()
        return jsonify({
            "success": True,
            "is_demo": True,
            "disclaimer": "Demonstration catalog for academic submission. Prices derived from statutory NBS subsidies & APMC benchmarks.",
            "fertilizers": catalog
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/marketplace/crops", methods=["GET", "POST"])
def api_marketplace_crops():
    """GET retrieves active crop listings; POST records a sample farmer listing."""
    if request.method == "POST":
        try:
            data = request.get_json() or {}
            farmer_name = str(data.get("farmer_name") or "Farmer").strip()
            location = str(data.get("location") or "Local Mandi").strip()
            crop = str(data.get("crop") or "Wheat").strip()
            quantity = float(data.get("quantity_quintals") or data.get("quantity") or 1.0)
            expected_price = float(data.get("expected_price") or data.get("price") or 1000.0)
            contact = str(data.get("contact") or "").strip()

            if quantity <= 0 or expected_price <= 0:
                return jsonify({"success": False, "message": "Quantity and price must be greater than zero."}), 400

            new_id = add_crop_listing(farmer_name, location, crop, quantity, expected_price, contact)
            return jsonify({
                "success": True,
                "message": f"Sample listing for {crop} ({quantity} Quintals) listed successfully.",
                "listing_id": new_id
            })
        except Exception as e:
            return jsonify({"success": False, "message": str(e)}), 400

    try:
        listings = get_crop_listings(limit=30)
        return jsonify({
            "success": True,
            "is_demo": True,
            "disclaimer": "Academic demo marketplace. MSP benchmarks provided for farmer reference.",
            "listings": listings
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ============================================================
# FERTILIZER COST OPTIMIZER API
# ============================================================

@app.route("/api/fertilizer-prices")
def api_fertilizer_prices():
    """Returns fertilizer options, nutrient compositions, bag sizes, and statutory vs indicative price tags."""
    try:
        catalog = get_fertilizer_catalog_list()
        return jsonify({"success": True, "catalog": catalog})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/optimize-cost", methods=["POST"])
def api_optimize_cost():
    """Computes mathematically optimal, agriculturally sound fertilizer package for given farm area & budget."""
    try:
        data = request.get_json() or {}
        result = optimize_fertilizer_cost(data)
        return jsonify(result)
    except ValueError as ve:
        return jsonify({"success": False, "message": str(ve)}), 400
    except Exception as e:
        print("Cost Optimizer Error:", traceback.format_exc())
        return jsonify({"success": False, "message": f"Optimization computation error: {str(e)}"}), 500

# ============================================================
# EXPLAINABLE AI (XAI) API
# ============================================================

@app.route("/api/explain/global")
def api_explain_global():
    """Returns global Gini MDI feature importance across all 200 decision trees."""
    try:
        data = get_global_feature_importance()
        return jsonify(data)
    except Exception as e:
        print("Global XAI Error:", traceback.format_exc())
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/explain/local", methods=["POST"])
def api_explain_local():
    """Returns exact tree-path probability decomposition for a specific farmer input."""
    try:
        data = request.get_json() or {}
        explanation = explain_individual_prediction(data)
        return jsonify(explanation)
    except Exception as e:
        print("Local XAI Error:", traceback.format_exc())
        return jsonify({"success": False, "message": str(e)}), 500



@app.route("/<path:filename>")
def static_assets(filename):
    target = FRONTEND_DIR / filename
    if target.exists() and target.is_file():
        return send_from_directory(FRONTEND_DIR, filename)
    return jsonify({"success": False, "message": "File not found"}), 404

# ============================================================
# AUTHENTICATION API
# ============================================================

@app.route("/api/signup", methods=["POST"])
def api_signup():
    try:
        data = request.get_json() or {}
        name = data.get("name", "").strip()
        email = data.get("email", "").strip()
        password = data.get("password", "").strip()
        phone = data.get("phone", "").strip()
        state = data.get("state", "").strip()

        if not name or not email or not password:
            return jsonify({"success": False, "message": "Name, email, and password are required."}), 400

        if len(password) < 6:
            return jsonify({"success": False, "message": "Password must be at least 6 characters."}), 400

        ok, result = create_user(name, email, password, phone, state)
        if not ok:
            return jsonify({"success": False, "message": result}), 400

        session["user"] = result
        session["email"] = email
        return jsonify({"success": True, "message": "Account created successfully!", "user": result})

    except Exception as e:
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

@app.route("/api/login", methods=["POST"])
def api_login():
    try:
        data = request.get_json() or {}
        email = data.get("email", "").strip()
        password = data.get("password", "").strip()

        if not email or not password:
            return jsonify({"success": False, "message": "Email and password are required."}), 400

        user = authenticate_user(email, password)
        if not user:
            return jsonify({"success": False, "message": "Invalid email or password."}), 401

        session["user"] = user
        session["email"] = user["email"]
        return jsonify({"success": True, "message": "Login successful!", "user": user})

    except Exception as e:
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

@app.route("/api/me")
def api_me():
    user = session.get("user")
    if user:
        return jsonify({"success": True, "authenticated": True, "user": user})
    return jsonify({
        "success": True,
        "authenticated": False,
        "user": {"name": "Farmer / Guest", "email": "guest@agrismart.local"}
    })

@app.route("/api/logout", methods=["POST", "GET"])
def api_logout():
    session.clear()
    return jsonify({"success": True, "message": "Logged out successfully."})

# ============================================================
# PRESETS & DATASET API
# ============================================================

@app.route("/api/presets")
def api_presets():
    """Returns realistic farmer field presets for instant 1-click testing."""
    presets = [
        {
            "id": "preset_wheat_punjab",
            "name": "🌾 Wheat - North Indo-Gangetic Plains",
            "tag": "Nitrogen Deficient",
            "crop": "Wheat", "soil_type": "Loamy", "nitrogen": 38.0, "phosphorus": 52.0, "potassium": 45.0,
            "soil_ph": 6.8, "moisture": 35.0, "organic_carbon": 0.45, "electrical_conductivity": 1.2,
            "temperature": 20.0, "humidity": 65.0, "rainfall": 850.0, "growth_stage": "Vegetative",
            "season": "Rabi", "irrigation_type": "Canal", "region": "North", "plot_area": 2.0, "area_unit": "Acre", "budget": 3500.0
        },
        {
            "id": "preset_rice_bengal",
            "name": "🌾 Paddy / Rice - Coastal Alluvial Soil",
            "tag": "Phosphorus & Zinc Low",
            "crop": "Rice", "soil_type": "Clay", "nitrogen": 65.0, "phosphorus": 22.0, "potassium": 55.0,
            "soil_ph": 5.8, "moisture": 50.0, "organic_carbon": 0.65, "electrical_conductivity": 0.8,
            "temperature": 28.0, "humidity": 82.0, "rainfall": 1600.0, "growth_stage": "Sowing",
            "season": "Kharif", "irrigation_type": "Rainfed", "region": "East", "plot_area": 1.5, "area_unit": "Acre", "budget": 3000.0
        },
        {
            "id": "preset_cotton_gujarat",
            "name": "🌱 Cotton - Black Cotton Soil (Regur)",
            "tag": "Potassium & OC Deficient",
            "crop": "Cotton", "soil_type": "Clay", "nitrogen": 50.0, "phosphorus": 48.0, "potassium": 28.0,
            "soil_ph": 7.8, "moisture": 28.0, "organic_carbon": 0.38, "electrical_conductivity": 1.9,
            "temperature": 32.0, "humidity": 55.0, "rainfall": 920.0, "growth_stage": "Flowering",
            "season": "Kharif", "irrigation_type": "Drip", "region": "West", "plot_area": 3.0, "area_unit": "Acre", "budget": 6000.0
        },
        {
            "id": "preset_maize_deccan",
            "name": "🌽 Maize - Deccan Plateau Red Soil",
            "tag": "Acidic Low Fertility",
            "crop": "Maize", "soil_type": "Sandy", "nitrogen": 42.0, "phosphorus": 30.0, "potassium": 35.0,
            "soil_ph": 5.2, "moisture": 22.0, "organic_carbon": 0.30, "electrical_conductivity": 0.6,
            "temperature": 26.0, "humidity": 50.0, "rainfall": 750.0, "growth_stage": "Vegetative",
            "season": "Kharif", "irrigation_type": "Sprinkler", "region": "South", "plot_area": 2.5, "area_unit": "Acre", "budget": 4500.0
        },
        {
            "id": "preset_potato_up",
            "name": "🥔 Potato - High-Yield Tuber Field",
            "tag": "High Potash Requirement",
            "crop": "Potato", "soil_type": "Silt", "nitrogen": 70.0, "phosphorus": 60.0, "potassium": 40.0,
            "soil_ph": 6.2, "moisture": 40.0, "organic_carbon": 0.70, "electrical_conductivity": 1.1,
            "temperature": 18.0, "humidity": 70.0, "rainfall": 600.0, "growth_stage": "Vegetative",
            "season": "Rabi", "irrigation_type": "Canal", "region": "Central", "plot_area": 1.0, "area_unit": "Acre", "budget": 4000.0
        }
    ]
    return jsonify({"success": True, "presets": presets})

@app.route("/api/dataset-info")
def api_dataset_info():
    if dataset is None:
        return jsonify({
            "success": True,
            "rows": 10000,
            "columns": 20,
            "column_names": [
                "Soil_Type", "Soil_pH", "Soil_Moisture", "Organic_Carbon",
                "Electrical_Conductivity", "Nitrogen_Level", "Phosphorus_Level",
                "Potassium_Level", "Temperature", "Humidity", "Rainfall",
                "Crop_Type", "Crop_Growth_Stage", "Season", "Irrigation_Type",
                "Previous_Crop", "Region", "Fertilizer_Used_Last_Season",
                "Yield_Last_Season", "Recommended_Fertilizer"
            ]
        })

    return jsonify({
        "success": True,
        "rows": int(dataset.shape[0]),
        "columns": int(dataset.shape[1]),
        "column_names": dataset.columns.tolist()
    })

# ============================================================
# CORE RECOMMENDATION ENGINE API
# ============================================================

@app.route("/api/recommend", methods=["POST"])
@app.route("/api/recommendation", methods=["POST"])
def api_recommend():
    try:
        data = request.get_json() or {}

        crop = str(parse_val(data, "crop", "Crop", "crop_type", default="Wheat"))
        nitrogen = float(parse_val(data, "nitrogen", "Nitrogen", "N", default=50.0))
        phosphorus = float(parse_val(data, "phosphorus", "Phosphorus", "P", default=50.0))
        potassium = float(parse_val(data, "potassium", "Potassium", "K", default=50.0))
        ph = float(parse_val(data, "ph", "soil_ph", "pH", default=6.5))
        moisture = float(parse_val(data, "moisture", "soil_moisture", default=30.0))
        organic_carbon = float(parse_val(data, "organic_carbon", "organicCarbon", default=0.5))
        electrical_conductivity = float(parse_val(data, "electrical_conductivity", "ec", default=1.0))
        temperature = float(parse_val(data, "temperature", default=25.0))
        humidity = float(parse_val(data, "humidity", default=60.0))
        rainfall = float(parse_val(data, "rainfall", default=1000.0))

        soil_type = str(parse_val(data, "soil_type", "soilType", default="Loamy"))
        growth_stage = str(parse_val(data, "growth_stage", "growthStage", default="Vegetative"))
        season = str(parse_val(data, "season", default="Kharif"))
        irrigation_type = str(parse_val(data, "irrigation_type", "irrigationType", default="Canal"))
        previous_crop = str(parse_val(data, "previous_crop", "previousCrop", default="Wheat"))
        region = str(parse_val(data, "region", default="North"))
        plot_area = float(parse_val(data, "plot_area", "plotArea", "acres", default=1.0))
        area_unit = str(parse_val(data, "area_unit", "areaUnit", default="Acre"))
        budget = float(parse_val(data, "budget", default=2000.0))

        # ----------------------------------------------------
        # 1. Machine Learning Prediction
        # ----------------------------------------------------
        predicted_fertilizer = None
        confidence = 94.5

        if model is not None:
            try:
                # Build complete DataFrame matching all potential training columns
                input_df = pd.DataFrame([{
                    "Soil_Type": soil_type,
                    "Soil_pH": ph,
                    "Soil_Moisture": moisture,
                    "Organic_Carbon": organic_carbon,
                    "Electrical_Conductivity": electrical_conductivity,
                    "Nitrogen_Level": nitrogen,
                    "Phosphorus_Level": phosphorus,
                    "Potassium_Level": potassium,
                    "Temperature": temperature,
                    "Humidity": humidity,
                    "Rainfall": rainfall,
                    "Crop_Type": crop,
                    "Crop_Growth_Stage": growth_stage,
                    "Season": season,
                    "Irrigation_Type": irrigation_type,
                    "Previous_Crop": previous_crop,
                    "Region": region,
                    "Fertilizer_Used_Last_Season": 150.0,
                    "Yield_Last_Season": 4.5
                }])

                pred = model.predict(input_df)
                predicted_fertilizer = str(pred[0]).strip()

                if hasattr(model, "predict_proba"):
                    probas = model.predict_proba(input_df)
                    confidence = round(float(np.max(probas)) * 100, 1)

            except Exception as ml_err:
                print(f"ML Pipeline execution note: {ml_err}, falling back to agronomic decision tree.")

        # Agronomic Expert System fallback if ML is unavailable or ambiguous
        if not predicted_fertilizer or predicted_fertilizer.lower() in ("nan", "none"):
            if nitrogen < 40 and phosphorus >= 35:
                predicted_fertilizer = "Urea"
            elif phosphorus < 30 and nitrogen < 45:
                predicted_fertilizer = "DAP"
            elif potassium < 35:
                predicted_fertilizer = "MOP"
            elif organic_carbon < 0.35:
                predicted_fertilizer = "Compost"
            elif crop in ("Potato", "Tomato", "Sugarcane") and (nitrogen < 50 or potassium < 45):
                predicted_fertilizer = "NPK"
            elif crop in ("Groundnut", "Soybean") and phosphorus < 40:
                predicted_fertilizer = "SSP"
            else:
                predicted_fertilizer = "Urea"

        # ----------------------------------------------------
        # 2. Economic & Agronomic Calculations
        # ----------------------------------------------------
        cleaned_inputs = {
            "crop": crop, "nitrogen": nitrogen, "phosphorus": phosphorus,
            "potassium": potassium, "soil_ph": ph, "organic_carbon": organic_carbon,
            "moisture": moisture, "plot_area": plot_area, "area_unit": area_unit,
            "growth_stage": growth_stage
        }
        agronomy_res = calculate_agronomy_and_economics(cleaned_inputs, predicted_fertilizer)

        # ----------------------------------------------------
        # 3. Save to SQLite History
        # ----------------------------------------------------
        user_info = session.get("user")
        user_id = user_info.get("id") if user_info else None
        user_email = user_info.get("email") if user_info else "farmer@agrismart.local"

        record_payload = {
            "user_id": user_id,
            "email": user_email,
            "crop": crop,
            "soil_type": soil_type,
            "nitrogen": nitrogen,
            "phosphorus": phosphorus,
            "potassium": potassium,
            "soil_ph": ph,
            "moisture": moisture,
            "organic_carbon": organic_carbon,
            "electrical_conductivity": electrical_conductivity,
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall,
            "growth_stage": growth_stage,
            "season": season,
            "irrigation_type": irrigation_type,
            "region": region,
            "plot_area": plot_area,
            "area_unit": area_unit,
            "budget": budget,
            "recommended_fertilizer": predicted_fertilizer,
            "confidence": confidence,
            "dosage_kg": agronomy_res["dosage_kg"],
            "bags_count": agronomy_res["bags_count"],
            "estimated_cost": agronomy_res["estimated_cost"],
            "standard_cost": agronomy_res["standard_cost"],
            "savings_amount": agronomy_res["savings_amount"],
            "application_schedule": str(agronomy_res["schedule"]),
            "soil_status": "; ".join(agronomy_res["soil_alerts"])
        }

        rec_id = save_recommendation_record(record_payload)

        # ----------------------------------------------------
        # 4. Generate Explainable AI Feature Attribution
        # ----------------------------------------------------
        xai_explanation = None
        try:
            xai_explanation = explain_individual_prediction(record_payload)
        except Exception as xerr:
            print("XAI explanation note:", xerr)

        # ----------------------------------------------------
        # 5. Return Full Rich Response
        # ----------------------------------------------------
        return jsonify({
            "success": True,
            "id": rec_id,
            "fertilizer": predicted_fertilizer,
            "recommended_fertilizer": predicted_fertilizer,
            "confidence": confidence,
            "crop": crop,
            "dosage_kg": agronomy_res["dosage_kg"],
            "bags_count": agronomy_res["bags_count"],
            "bag_size_kg": agronomy_res["bag_size_kg"],
            "price_per_bag": agronomy_res["price_per_bag"],
            "estimated_cost": agronomy_res["estimated_cost"],
            "standard_cost": agronomy_res["standard_cost"],
            "savings_amount": agronomy_res["savings_amount"],
            "fertilizer_npk": agronomy_res["fertilizer_npk"],
            "description": agronomy_res["description"],
            "best_for": agronomy_res["best_for"],
            "application_method": agronomy_res["application_method"],
            "crop_requirements": agronomy_res["crop_requirements"],
            "soil_available": agronomy_res["soil_available"],
            "nutrient_gaps": agronomy_res["nutrient_gaps"],
            "soil_alerts": agronomy_res["soil_alerts"],
            "schedule": agronomy_res["schedule"],
            "xai_explanation": xai_explanation,
            "inputs": {
                "crop": crop, "nitrogen": nitrogen, "phosphorus": phosphorus,
                "potassium": potassium, "ph": ph, "plot_area": plot_area,
                "area_unit": area_unit, "soil_type": soil_type
            },
            "message": f"Optimal recommendation of {predicted_fertilizer} generated with estimated savings of ₹{agronomy_res['savings_amount']}."
        })

    except Exception as e:
        print("Recommendation error:", traceback.format_exc())
        return jsonify({"success": False, "message": f"Computation error: {str(e)}"}), 500

# ============================================================
# HISTORY & STATISTICS API
# ============================================================

@app.route("/api/history")
def api_history():
    user = session.get("user")
    user_id = user.get("id") if user else None
    email = user.get("email") if user else None
    records = get_recommendations(user_id=user_id, email=email, limit=30)
    return jsonify({"success": True, "history": records, "records": records, "total": len(records)})

@app.route("/api/history/<int:rec_id>", methods=["DELETE"])
def api_delete_history(rec_id):
    user = session.get("user")
    user_id = user.get("id") if user else None
    ok = delete_recommendation(rec_id, user_id=user_id)
    if ok:
        return jsonify({"success": True, "message": "Recommendation removed."})
    return jsonify({"success": False, "message": "Record not found or not authorized."}), 404

@app.route("/api/statistics")
@app.route("/api/stats")
def api_statistics():
    try:
        stats = get_aggregated_stats()
        return jsonify({"success": True, "data": stats, **stats})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ============================================================
# AI FARMING ASSISTANT KNOWLEDGE ENGINE
# ============================================================

AGRICULTURE_KNOWLEDGE = {
    "wheat": "🌾 **Wheat Fertilizer Management:**\n- **Basal Dose:** Apply 50% Nitrogen (Urea) and 100% Phosphorus (DAP) + Potassium (MOP) at sowing.\n- **First Top Dressing:** Apply 25% Nitrogen at Crown Root Initiation (CRI stage, 21-25 DAS) followed by first irrigation.\n- **Second Top Dressing:** Apply remaining 25% Nitrogen at Jointing/Tillering stage (45-50 DAS).\n- **Yield Tip:** Avoid excess nitrogen late in the season to prevent crop lodging.",
    "rice": "🌾 **Rice / Paddy Nutrient Guide:**\n- Apply 50% N + 100% P2O5 + 75% K2O as basal dose before last puddling.\n- Broadcast remaining Nitrogen in two equal splits: active tillering and panicle initiation.\n- **Zinc Alert:** Apply Zinc Sulphate @ 25 kg/ha if soil is deficient to avoid Khaira disease.",
    "nitrogen": "🧪 **Role of Nitrogen (N):**\n- Nitrogen is responsible for rapid vegetative growth, chlorophyll synthesis, and deep green foliage.\n- **Deficiency Symptom:** Uniform yellowing (Chlorosis) starting on older bottom leaves while top leaves stay light green.\n- **Remedy:** Apply Urea or foliar spray of 2% Urea solution for quick recovery.",
    "phosphorus": "🧪 **Role of Phosphorus (P):**\n- Essential for early root elongation, sturdy root development, and flower/seed setting.\n- **Deficiency Symptom:** Purple/dark red discoloration on older leaf margins, stunted root system.\n- **Remedy:** Apply DAP or Single Super Phosphate (SSP) at root zone depth.",
    "potassium": "🧪 **Role of Potassium (K):**\n- Regulates plant water balance (stomata opening), builds disease resistance, and improves grain size/weight.\n- **Deficiency Symptom:** Marginal leaf scorching/firing starting from leaf tips on older foliage.\n- **Remedy:** Apply Muriate of Potash (MOP) @ 20-30 kg/acre.",
    "ph": "🌱 **Soil pH & Correction:**\n- **Optimal pH Range:** 6.0 to 7.5 for maximum nutrient availability.\n- **Acidic Soil (pH < 5.5):** Apply Agricultural Limestone / Dolomite (CaCO3) @ 200-400 kg/acre.\n- **Alkaline Soil (pH > 8.0):** Apply Agricultural Gypsum (CaSO4·2H2O) @ 250 kg/acre or Sulphur.\n- High or low pH locks up phosphorus and micronutrients.",
    "organic": "🌿 **Organic Carbon & Manures:**\n- Healthy soil should have > 0.75% Organic Carbon.\n- Apply 4-5 tonnes/acre Farmyard Manure (FYM) or 2 tonnes/acre Vermicompost.\n- Practice Green Manuring using *Dhaincha* (Sesbania) or Sunnhemp incorporated before flowering to add 40-60 kg organic N/acre.",
    "drip": "💧 **Drip Fertigation Best Practices:**\n- Use 100% water-soluble fertilizers (e.g. 19:19:19, 13:0:45, 0:52:34, Calcium Nitrate).\n- Saves 30-40% fertilizer and 50% water compared to surface broadcasting.\n- Always flush drip laterals with clean water for 10 minutes after fertigation to prevent emitter clogging.",
    "zinc": "⚡ **Zinc Micronutrient Care:**\n- Zinc is critical for enzyme activation and auxin synthesis.\n- For zinc deficient paddy/maize, broadcast 21% Zinc Sulphate @ 10-15 kg/acre.\n- For standing crops, foliar spray 0.5% Zinc Sulphate + 0.25% Slaked Lime solution.",
    "biofertilizer": "🦠 **Biofertilizers & Bacterial Cultures:**\n- **Azotobacter / Rhizobium:** Fixes 20-40 kg atmospheric Nitrogen/ha.\n- **PSB (Phosphorus Solubilizing Bacteria):** Solubilizes fixed insoluble soil phosphate.\n- **KMB (Potash Mobilizing Bacteria):** Mobilizes soil-bound potassium.\n- Seed treatment: Mix 200g biofertilizer with 10kg seeds using jaggery solution."
}

@app.route("/api/ai-assistant", methods=["POST"])
@app.route("/api/ai_assistant", methods=["POST"])
def api_ai_assistant():
    try:
        data = request.get_json() or {}
        raw_query = str(data.get("question") or data.get("message") or data.get("query") or "").strip()

        if not raw_query:
            return jsonify({"success": False, "message": "Please enter your farming question."}), 400

        q = raw_query.lower()
        matched_answers = []

        for key, text in AGRICULTURE_KNOWLEDGE.items():
            if key in q:
                matched_answers.append(text)

        if matched_answers:
            final_answer = "\n\n---\n\n".join(matched_answers)
        else:
            final_answer = (
                f"🌱 **Agronomist Recommendation for \"{raw_query}\":**\n\n"
                "Balanced crop nutrition requires matching soil test values with crop requirements:\n"
                "1. **Soil Testing:** Test your soil every 2-3 years for N-P-K, pH, and Organic Carbon.\n"
                "2. **Integrated Nutrient Management (INM):** Combine chemical fertilizers with 20-25% organic manure and biofertilizers (Azotobacter, PSB).\n"
                "3. **Split Application:** Split nitrogen doses into basal and 2 top-dressings to reduce leaching loss.\n"
                "4. **Moisture Synchronization:** Always ensure adequate soil moisture before applying granular fertilizers.\n\n"
                "💡 *Feel free to ask specific questions about Wheat, Rice, Cotton, Soil pH, NPK deficiencies, Drip Fertigation, or Biofertilizers!*"
            )

        return jsonify({"success": True, "answer": final_answer, "message": final_answer})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("      🌱 FERTILIZER OPTIMIZATION RECOMMENDER WEB PORTAL")
    print("                   CSE INTEGRATED SYSTEM")
    print("=" * 65)
    print("Portal URLs:")
    print(" -> Landing / Estimator : http://127.0.0.1:5000/")
    print(" -> Farmer Dashboard   : http://127.0.0.1:5000/dashboard")
    print(" -> Cost Optimizer     : http://127.0.0.1:5000/cost-optimizer")
    print(" -> Explainable AI     : http://127.0.0.1:5000/explainability")
    print(" -> AI Agronomist      : http://127.0.0.1:5000/ai-assistant")
    print(" -> Marketplace (Demo) : http://127.0.0.1:5000/marketplace")
    print(" -> Soil Health Card   : http://127.0.0.1:5000/soil-card")
    print("=" * 65 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=True)