"""
AgriSmart - Explainable AI (XAI) Engine for Fertilizer Recommendation
Provides mathematically sound global and local feature explanations for the Random Forest pipeline:
1. Global Feature Importance: Mean Decrease in Impurity (Gini MDI) mapped back to original agronomic features.
2. Local Feature Explanation: Exact Decision-Tree Path Decomposition (TreeInterpreter) showing positive/negative
   feature contributions to the predicted class probability.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "fertilizer_model.pkl"

_CACHED_MODEL = None
_CACHED_GLOBAL_IMPORTANCE = None


def get_model():
    """Loads and caches the trained Random Forest pipeline."""
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        _CACHED_MODEL = joblib.load(MODEL_PATH)
    return _CACHED_MODEL


def get_global_feature_importance() -> Dict[str, Any]:
    """
    Computes global feature importance across all 200 decision trees.
    Aggregates one-hot encoded dummy columns into their parent domain features
    (e.g., Crop_Type, Soil_Type, Nitrogen_Level, Soil_pH).
    """
    global _CACHED_GLOBAL_IMPORTANCE
    if _CACHED_GLOBAL_IMPORTANCE is not None:
        return _CACHED_GLOBAL_IMPORTANCE

    pipeline = get_model()
    preprocessor = pipeline.named_steps["preprocessor"]
    rf = pipeline.named_steps["model"]

    raw_features = preprocessor.get_feature_names_out()
    raw_importances = rf.feature_importances_

    # Map raw encoded columns to original high-level domain features
    # Format of raw_features: "categorical__Crop_Type_Wheat" or "numerical__Nitrogen_Level"
    domain_importances: Dict[str, float] = {}
    encoded_feature_details: List[Dict[str, Any]] = []

    for name, imp in zip(raw_features, raw_importances):
        clean_name = name
        if "__" in clean_name:
            clean_name = clean_name.split("__", 1)[1]

        # Determine parent feature
        if "_" in clean_name and any(clean_name.startswith(p) for p in [
            "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
            "Irrigation_Type", "Previous_Crop", "Region"
        ]):
            for p in ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season", "Irrigation_Type", "Previous_Crop", "Region"]:
                if clean_name.startswith(p):
                    parent = p
                    break
            else:
                parent = clean_name
        else:
            parent = clean_name

        domain_importances[parent] = domain_importances.get(parent, 0.0) + float(imp)

        encoded_feature_details.append({
            "encoded_name": name,
            "display_name": clean_name.replace("_", " "),
            "importance": round(float(imp), 4),
            "percentage": round(float(imp) * 100, 2)
        })

    # Sort high-level features
    sorted_domain = sorted(domain_importances.items(), key=lambda x: x[1], reverse=True)
    total_imp = sum(domain_importances.values()) or 1.0

    domain_summary = []
    for rank, (feat, val) in enumerate(sorted_domain, 1):
        domain_summary.append({
            "rank": rank,
            "feature": feat,
            "display_name": feat.replace("_", " "),
            "importance": round(val, 4),
            "percentage": round((val / total_imp) * 100, 2),
            "description": get_feature_description(feat)
        })

    encoded_feature_details.sort(key=lambda x: x["importance"], reverse=True)

    _CACHED_GLOBAL_IMPORTANCE = {
        "success": True,
        "method": "Mean Decrease in Impurity (Gini MDI across 200 Decision Trees)",
        "explanation_type": "Global Model Architecture Feature Importance",
        "total_features": len(domain_summary),
        "features": domain_summary,
        "top_features": domain_summary[:8],
        "encoded_features": encoded_feature_details[:15],
        "scientific_note": (
            "Global feature importance measures the average reduction in Gini impurity brought by each "
            "feature across all node splits in the Random Forest. It reveals which variables the model relies on "
            "most across the entire historical training population, not on an individual farmer's sample."
        )
    }
    return _CACHED_GLOBAL_IMPORTANCE


def explain_individual_prediction(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decomposes an individual farmer prediction into exact feature contributions
    using Decision Tree Path Probability Decomposition (TreeInterpreter method).
    
    Guarantees:
      Predicted Probability = Baseline Prior Rate + Sum of Feature Contributions
    """
    pipeline = get_model()
    preprocessor = pipeline.named_steps["preprocessor"]
    rf = pipeline.named_steps["model"]

    # Format input into standard 19-column DataFrame
    input_df = pd.DataFrame([{
        "Soil_Type": str(input_data.get("soil_type", "Loamy")),
        "Soil_pH": float(input_data.get("soil_ph", input_data.get("ph", 6.5))),
        "Soil_Moisture": float(input_data.get("moisture", 30.0)),
        "Organic_Carbon": float(input_data.get("organic_carbon", 0.5)),
        "Electrical_Conductivity": float(input_data.get("electrical_conductivity", input_data.get("ec", 1.0))),
        "Nitrogen_Level": float(input_data.get("nitrogen", 50.0)),
        "Phosphorus_Level": float(input_data.get("phosphorus", 50.0)),
        "Potassium_Level": float(input_data.get("potassium", 50.0)),
        "Temperature": float(input_data.get("temperature", 25.0)),
        "Humidity": float(input_data.get("humidity", 60.0)),
        "Rainfall": float(input_data.get("rainfall", 1000.0)),
        "Crop_Type": str(input_data.get("crop", "Wheat")),
        "Crop_Growth_Stage": str(input_data.get("growth_stage", "Vegetative")),
        "Season": str(input_data.get("season", "Kharif")),
        "Irrigation_Type": str(input_data.get("irrigation_type", "Canal")),
        "Previous_Crop": str(input_data.get("previous_crop", "Wheat")),
        "Region": str(input_data.get("region", "North")),
        "Fertilizer_Used_Last_Season": float(input_data.get("fertilizer_used_last_season", 150.0)),
        "Yield_Last_Season": float(input_data.get("yield_last_season", 4.5))
    }])

    # Preprocess
    Xt = preprocessor.transform(input_df)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()

    classes = rf.classes_
    n_classes = len(classes)
    n_features = Xt.shape[1]

    # Predict probabilities and target class
    probas = rf.predict_proba(Xt)[0]
    pred_idx = int(np.argmax(probas))
    predicted_fertilizer = str(classes[pred_idx])
    confidence = float(probas[pred_idx])

    # Decompose decision paths across all estimators
    n_trees = len(rf.estimators_)
    biases = np.zeros(n_classes)
    contributions = np.zeros((n_features, n_classes))

    for tree in rf.estimators_:
        t = tree.tree_
        # Root node class distribution
        root_val = t.value[0, 0] / np.sum(t.value[0, 0])
        biases += root_val

        # Get nodes traversed by this sample
        dp = tree.decision_path(Xt)
        path = dp.indices

        for i in range(len(path) - 1):
            cur_node = path[i]
            next_node = path[i + 1]
            split_feature = t.feature[cur_node]

            v_cur = t.value[cur_node, 0] / np.sum(t.value[cur_node, 0])
            v_next = t.value[next_node, 0] / np.sum(t.value[next_node, 0])

            contributions[split_feature] += (v_next - v_cur)

    biases /= n_trees
    contributions /= n_trees

    base_probability = float(biases[pred_idx])
    target_contributions = contributions[:, pred_idx]

    # Map back to high-level domain features
    raw_features = preprocessor.get_feature_names_out()
    domain_contributions: Dict[str, float] = {}

    for name, c_val in zip(raw_features, target_contributions):
        clean_name = name.split("__", 1)[1] if "__" in name else name

        for p in ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season", "Irrigation_Type", "Previous_Crop", "Region"]:
            if clean_name.startswith(p):
                parent = p
                break
        else:
            parent = clean_name

        domain_contributions[parent] = domain_contributions.get(parent, 0.0) + float(c_val)

    # Sort contributions into positive and negative drivers
    positive_drivers = []
    negative_drivers = []

    for feat, val in sorted(domain_contributions.items(), key=lambda x: abs(x[1]), reverse=True):
        entry = {
            "feature": feat,
            "display_name": feat.replace("_", " "),
            "contribution": round(val, 4),
            "percentage_impact": round(val * 100, 2),
            "direction": "Pushed towards prediction" if val > 0 else "Pushed against prediction",
            "current_value": get_human_value(input_df.iloc[0], feat)
        }
        if val > 0:
            positive_drivers.append(entry)
        elif val < 0:
            negative_drivers.append(entry)

    # Construct plain-language narrative
    top_pos_names = [f"**{p['display_name']}** ({p['current_value']})" for p in positive_drivers[:3]]
    narrative_pos = ", ".join(top_pos_names) if top_pos_names else "Balanced baseline characteristics"

    explanation_narrative = (
        f"The Random Forest model predicted **{predicted_fertilizer}** with **{round(confidence * 100, 1)}%** confidence. "
        f"Before considering field specifics, the model has a general baseline prior probability of {round(base_probability * 100, 1)}% for this fertilizer. "
        f"For this specific soil test, the strongest factors increasing the prediction probability were {narrative_pos}."
    )

    return {
        "success": True,
        "method": "Exact Tree-Path Probability Decomposition (TreeInterpreter)",
        "explanation_type": "Local Instance Feature Attribution",
        "predicted_fertilizer": predicted_fertilizer,
        "confidence": round(confidence * 100, 2),
        "base_probability": round(base_probability * 100, 2),
        "narrative": explanation_narrative,
        "positive_drivers": positive_drivers[:6],
        "negative_drivers": negative_drivers[:4],
        "all_feature_contributions": [
            {
                "feature": k,
                "display_name": k.replace("_", " "),
                "contribution": round(v, 4),
                "impact_percent": round(v * 100, 2),
                "current_value": get_human_value(input_df.iloc[0], k)
            }
            for k, v in sorted(domain_contributions.items(), key=lambda x: x[1], reverse=True)
        ],
        "disclaimer": (
            "Notice: These feature contributions reflect statistical patterns and decision boundaries "
            "learned by the Random Forest trees from the training dataset. They explain model prediction "
            "behavior, not direct biological or agronomic causation."
        )
    }


def get_feature_description(feature: str) -> str:
    """Returns educational descriptions of agronomic features."""
    descriptions = {
        "Nitrogen_Level": "Primary macronutrient driving leafy foliage and chlorophyll synthesis.",
        "Phosphorus_Level": "Macronutrient vital for early root expansion, flower initiation, and seed vigor.",
        "Potassium_Level": "Macronutrient regulating stomatal water balance, disease resistance, and stem strength.",
        "Soil_pH": "Determines soil acidity/alkalinity and controls bioavailability of macro and micro nutrients.",
        "Organic_Carbon": "Indicator of microbial biomass, soil structure, and natural moisture holding capacity.",
        "Soil_Moisture": "Percentage water in soil pore space necessary for nutrient dissolution and root uptake.",
        "Electrical_Conductivity": "Measure of soluble salt concentration (salinity indicator).",
        "Crop_Type": "Plant species defining total stoichiometric nutrient uptake targets.",
        "Soil_Type": "Soil physical texture (Sand, Silt, Clay, Loam) affecting nutrient leaching and retention.",
        "Crop_Growth_Stage": "Phenological stage (Sowing, Vegetative, Flowering, Harvest) determining uptake timing.",
        "Season": "Agricultural cropping window (Kharif, Rabi, Zaid) defining temperature and day-length regime.",
        "Temperature": "Ambient temperature influencing transpiration and metabolic rate.",
        "Humidity": "Atmospheric moisture affecting crop water demand.",
        "Rainfall": "Seasonal precipitation affecting leaching and moisture availability.",
        "Irrigation_Type": "Water delivery method (Canal, Drip, Sprinkler, Rainfed).",
        "Region": "Geographic agro-climatic zone.",
        "Previous_Crop": "Preceding crop influencing residual nitrogen and organic residue.",
        "Fertilizer_Used_Last_Season": "Historical chemical fertilizer volume.",
        "Yield_Last_Season": "Historical productivity benchmark."
    }
    return descriptions.get(feature, "Agronomic input variable.")


def get_human_value(row: pd.Series, feature: str) -> str:
    """Formats current input value into a clean human-readable string."""
    if feature not in row:
        return ""
    val = row[feature]
    if isinstance(val, (int, float, np.number)):
        return f"{val:.1f}"
    return str(val)
