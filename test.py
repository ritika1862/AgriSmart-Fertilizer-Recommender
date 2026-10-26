"""
AgriSmart - Automated Test Suite
Tests dataset integrity, ML pipeline, agronomy calculations, database operations, and Flask REST endpoints.
"""

import unittest
import os
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Set base directory to project root
BASE_DIR = Path(__file__).resolve().parent

# Import project modules
from database.database import (
    hash_password,
    get_connection,
    save_recommendation_record,
    get_recommendations,
    delete_recommendation,
    get_aggregated_stats,
)
from backend.main import (
    app,
    calculate_agronomy_and_economics,
    FERTILIZER_MASTER,
    CROP_STANDARDS,
)


class TestDataIntegrity(unittest.TestCase):
    """Verifies that dataset files exist and meet schema expectations."""

    def test_fertilizer_recommendation_csv_exists_and_valid(self):
        dataset_path = BASE_DIR / "data" / "fertilizer_recommendation.csv"
        self.assertTrue(dataset_path.exists(), "fertilizer_recommendation.csv is missing!")
        df = pd.read_csv(dataset_path)
        self.assertGreaterEqual(len(df), 1000, "Dataset should have at least 1,000 records.")
        self.assertIn("Recommended_Fertilizer", df.columns, "Target column is missing from CSV.")
        self.assertIn("Crop_Type", df.columns, "Crop_Type column is missing.")
        self.assertIn("Nitrogen_Level", df.columns, "Nitrogen_Level column is missing.")

    def test_fertilizer_master_csv_exists_and_valid(self):
        master_path = BASE_DIR / "data" / "fertilizer_master.csv"
        self.assertTrue(master_path.exists(), "fertilizer_master.csv is missing!")
        df = pd.read_csv(master_path)
        self.assertIn("Fertilizer", df.columns)
        self.assertIn("Package_kg", df.columns)
        self.assertIn("Price_INR", df.columns)
        fertilizers = df["Fertilizer"].tolist()
        for expected in ["Urea", "DAP", "MOP"]:
            self.assertIn(expected, fertilizers, f"Expected fertilizer {expected} in master CSV.")


class TestMachineLearningPipeline(unittest.TestCase):
    """Verifies the serialized Random Forest pipeline loads and generates valid predictions."""

    def setUp(self):
        self.model_path = BASE_DIR / "fertilizer_model.pkl"

    def test_model_artifact_exists_and_loads(self):
        self.assertTrue(self.model_path.exists(), "fertilizer_model.pkl is missing!")
        model = joblib.load(self.model_path)
        self.assertTrue(hasattr(model, "predict"), "Loaded model has no predict method.")
        self.assertTrue(hasattr(model, "named_steps"), "Loaded model is not a valid Pipeline.")
        self.assertIn("preprocessor", model.named_steps)
        self.assertIn("model", model.named_steps)

    def test_model_prediction_and_confidence(self):
        model = joblib.load(self.model_path)
        sample_input = pd.DataFrame([{
            "Soil_Type": "Loamy",
            "Soil_pH": 6.5,
            "Soil_Moisture": 35.0,
            "Organic_Carbon": 0.55,
            "Electrical_Conductivity": 1.1,
            "Nitrogen_Level": 50.0,
            "Phosphorus_Level": 45.0,
            "Potassium_Level": 40.0,
            "Temperature": 25.0,
            "Humidity": 65.0,
            "Rainfall": 950.0,
            "Crop_Type": "Wheat",
            "Crop_Growth_Stage": "Vegetative",
            "Season": "Rabi",
            "Irrigation_Type": "Canal",
            "Previous_Crop": "Rice",
            "Region": "North",
            "Fertilizer_Used_Last_Season": 150.0,
            "Yield_Last_Season": 4.5
        }])

        pred = model.predict(sample_input)
        self.assertEqual(len(pred), 1)
        self.assertIsInstance(pred[0], str)
        self.assertTrue(len(pred[0]) > 0)

        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(sample_input)
            self.assertEqual(probas.shape[0], 1)
            self.assertGreaterEqual(float(np.max(probas)), 0.0)
            self.assertLessEqual(float(np.max(probas)), 1.0)


class TestAgronomyAndEconomics(unittest.TestCase):
    """Verifies nutrient gap, dosage, and cost savings calculations."""

    def test_wheat_calculation_logic(self):
        inputs = {
            "crop": "Wheat",
            "nitrogen": 40.0,
            "phosphorus": 30.0,
            "potassium": 25.0,
            "soil_ph": 6.5,
            "organic_carbon": 0.5,
            "moisture": 30.0,
            "plot_area": 1.0,
            "area_unit": "Acre",
            "growth_stage": "Vegetative"
        }
        res = calculate_agronomy_and_economics(inputs, "Urea")
        self.assertGreater(res["dosage_kg"], 0.0)
        self.assertGreaterEqual(res["bags_count"], 1)
        self.assertGreater(res["estimated_cost"], 0.0)
        self.assertGreaterEqual(res["savings_amount"], 0.0)
        self.assertEqual(len(res["schedule"]), 3)

    def test_soil_health_diagnostic_alerts(self):
        # Strongly acidic soil
        acidic_inputs = {"crop": "Rice", "soil_ph": 4.8, "organic_carbon": 0.3, "nitrogen": 20.0}
        res = calculate_agronomy_and_economics(acidic_inputs, "DAP")
        alert_text = " ".join(res["soil_alerts"])
        self.assertIn("Acidic", alert_text)
        self.assertIn("Organic Carbon", alert_text)
        self.assertIn("Nitrogen Deficit", alert_text)


class TestDatabaseModule(unittest.TestCase):
    """Verifies SQLite database operations, hashing, and record persistence."""

    def test_hash_password(self):
        hashed1 = hash_password("secret123")
        hashed2 = hash_password("secret123")
        self.assertEqual(hashed1, hashed2)
        self.assertEqual(len(hashed1), 64, "SHA-256 hash must be 64 hexadecimal characters.")

    def test_save_and_retrieve_recommendation(self):
        test_payload = {
            "user_id": None,
            "email": "test_runner@agrismart.local",
            "crop": "Maize",
            "soil_type": "Sandy",
            "nitrogen": 45.0,
            "phosphorus": 35.0,
            "potassium": 30.0,
            "soil_ph": 6.2,
            "plot_area": 1.0,
            "area_unit": "Acre",
            "recommended_fertilizer": "Urea",
            "confidence": 92.5,
            "dosage_kg": 50.0,
            "bags_count": 2.0,
            "estimated_cost": 533.0,
            "standard_cost": 1850.0,
            "savings_amount": 1317.0
        }
        rec_id = save_recommendation_record(test_payload)
        self.assertIsNotNone(rec_id)
        self.assertGreater(rec_id, 0)

        # Retrieve and verify
        recs = get_recommendations(email="test_runner@agrismart.local", limit=5)
        self.assertTrue(any(r["id"] == rec_id for r in recs))

        # Cleanup test record
        delete_recommendation(rec_id)

    def test_get_aggregated_stats(self):
        stats = get_aggregated_stats()
        self.assertIn("total_recommendations", stats)
        self.assertIn("fertilizer_distribution", stats)
        self.assertIn("crop_distribution", stats)
        self.assertIn("averages", stats)


class TestFlaskAPI(unittest.TestCase):
    """Verifies HTTP endpoints and REST API contracts."""

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_home_page_returns_200(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)

    def test_dashboard_page_returns_200(self):
        res = self.client.get("/dashboard")
        self.assertEqual(res.status_code, 200)

    def test_presets_api(self):
        res = self.client.get("/api/presets")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertIsInstance(data.get("presets"), list)
        self.assertGreaterEqual(len(data["presets"]), 1)

    def test_dataset_info_api(self):
        res = self.client.get("/api/dataset-info")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertIn("column_names", data)

    def test_recommend_endpoint_success(self):
        payload = {
            "crop": "Wheat",
            "soil_type": "Loamy",
            "nitrogen": 45.0,
            "phosphorus": 40.0,
            "potassium": 35.0,
            "ph": 6.8,
            "moisture": 30.0,
            "organic_carbon": 0.5,
            "electrical_conductivity": 1.0,
            "plot_area": 2.0,
            "area_unit": "Acre",
            "budget": 3000.0
        }
        res = self.client.post("/api/recommend", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertIn("fertilizer", data)
        self.assertIn("dosage_kg", data)
        self.assertIn("bags_count", data)
        self.assertIn("estimated_cost", data)
        self.assertIn("savings_amount", data)
        self.assertIn("schedule", data)

        # Cleanup created recommendation from database
        if data.get("id"):
            delete_recommendation(data["id"])

    def test_ai_assistant_endpoint(self):
        res = self.client.post("/api/ai-assistant", json={"question": "What is the best fertilizer for wheat?"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertIn("answer", data)
        self.assertIn("Wheat", data["answer"])


class TestFertilizerCostOptimizer(unittest.TestCase):
    """Verifies the Cost Optimizer engine, agricultural constraints, budget checking, and price transparency."""

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_optimizer_input_validation_missing_crop(self):
        from backend.optimizer import validate_optimizer_inputs
        with self.assertRaises(ValueError) as ctx:
            validate_optimizer_inputs({"crop": "", "plot_area": 1.0})
        self.assertIn("Crop name is required", str(ctx.exception))

    def test_optimizer_input_validation_unsupported_crop(self):
        from backend.optimizer import validate_optimizer_inputs
        with self.assertRaises(ValueError) as ctx:
            validate_optimizer_inputs({"crop": "NonExistentCrop123", "plot_area": 1.0})
        self.assertIn("Verified ICAR nutrient standards are currently not available", str(ctx.exception))

    def test_optimizer_input_validation_negative_area(self):
        from backend.optimizer import validate_optimizer_inputs
        with self.assertRaises(ValueError) as ctx:
            validate_optimizer_inputs({"crop": "Wheat", "plot_area": -2.0})
        self.assertIn("greater than zero", str(ctx.exception))

    def test_optimizer_input_validation_negative_budget(self):
        from backend.optimizer import validate_optimizer_inputs
        with self.assertRaises(ValueError) as ctx:
            validate_optimizer_inputs({"crop": "Wheat", "plot_area": 1.0, "budget": -500})
        self.assertIn("Budget cannot be negative", str(ctx.exception))

    def test_price_transparency_labels(self):
        from backend.optimizer import get_fertilizer_catalog_list
        catalog = get_fertilizer_catalog_list()
        catalog_map = {item["key"]: item for item in catalog}

        self.assertIn("Urea", catalog_map)
        self.assertEqual(catalog_map["Urea"]["price_type"], "Statutory Subsidized MRP")
        self.assertEqual(catalog_map["Urea"]["price_per_bag"], 266.50)

        self.assertIn("DAP", catalog_map)
        self.assertEqual(catalog_map["DAP"]["price_type"], "Official Indicative Subsidy Benchmark")
        self.assertEqual(catalog_map["DAP"]["price_per_bag"], 1350.00)

        self.assertIn("MOP", catalog_map)
        self.assertEqual(catalog_map["MOP"]["price_type"], "Indicative Market Benchmark")

    def test_cost_optimization_satisfies_all_nutrient_deficits(self):
        from backend.optimizer import optimize_fertilizer_cost
        payload = {
            "crop": "Wheat",
            "plot_area": 2.0,
            "area_unit": "Acre",
            "budget": 10000.0,
            "nitrogen": 40.0,
            "phosphorus": 30.0,
            "potassium": 25.0
        }
        res = optimize_fertilizer_cost(payload)
        self.assertTrue(res["success"])
        opt = res["optimal_plan"]

        # Ensure optimal plan satisfies or exceeds the calculated deficit
        self.assertGreaterEqual(opt["supplied"]["n"], res["nutrient_gaps"]["n"])
        self.assertGreaterEqual(opt["supplied"]["p"], res["nutrient_gaps"]["p"])
        self.assertGreaterEqual(opt["supplied"]["k"], res["nutrient_gaps"]["k"])

        # Cost must be non-zero and savings must be computed
        self.assertGreater(opt["total_cost"], 0.0)
        self.assertGreaterEqual(opt["savings_amount"], 0.0)
        self.assertTrue(opt["within_budget"])
        self.assertEqual(len(res["all_strategies"]), 3)

    def test_budget_shortfall_detection(self):
        from backend.optimizer import optimize_fertilizer_cost
        # Extremely small budget of ₹100 for 5 acres of wheat
        payload = {
            "crop": "Wheat",
            "plot_area": 5.0,
            "area_unit": "Acre",
            "budget": 100.0,
            "nitrogen": 30.0,
            "phosphorus": 20.0,
            "potassium": 15.0
        }
        res = optimize_fertilizer_cost(payload)
        self.assertFalse(res["optimal_plan"]["within_budget"])
        self.assertGreater(res["optimal_plan"]["budget_deficit"], 0.0)
        self.assertTrue(any("Budget Shortfall" in w for w in res["warnings"]))

    def test_optimizer_page_route(self):
        res = self.client.get("/cost-optimizer")
        self.assertEqual(res.status_code, 200)

    def test_api_fertilizer_prices_endpoint(self):
        res = self.client.get("/api/fertilizer-prices")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertIsInstance(data.get("catalog"), list)
        self.assertGreaterEqual(len(data["catalog"]), 4)

    def test_api_optimize_cost_endpoint_success(self):
        payload = {
            "crop": "Rice",
            "plot_area": 1.5,
            "area_unit": "Acre",
            "budget": 5000.0,
            "nitrogen": 45.0,
            "phosphorus": 35.0,
            "potassium": 30.0
        }
        res = self.client.post("/api/optimize-cost", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertIn("optimal_plan", data)
        self.assertIn("all_strategies", data)
        self.assertGreater(data["optimal_plan"]["total_cost"], 0)

    def test_api_optimize_cost_endpoint_validation_error(self):
        payload = {
            "crop": "",
            "plot_area": -1.0
        }
        res = self.client.post("/api/optimize-cost", json=payload)
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data.get("success"))
        self.assertIn("message", data)

    def test_optimizer_positional_arguments_compatibility(self):
        from backend.optimizer import optimize_fertilizer_cost
        res = optimize_fertilizer_cost("Wheat", 2.0, "Acre", 3000.0)
        self.assertTrue(res["success"])
        self.assertGreater(res["total_cost"], 0)
        self.assertEqual(res["crop"], "Wheat")

    def test_recommend_page_route(self):
        res = self.client.get("/recommend")
        self.assertEqual(res.status_code, 200)


class TestExplainableAI(unittest.TestCase):
    """Verifies Global MDI Feature Importance and Local Tree-Path Probability Decomposition."""

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_global_feature_importance_structure(self):
        from backend.explainability import get_global_feature_importance
        res = get_global_feature_importance()
        self.assertTrue(res["success"])
        self.assertIn("features", res)
        self.assertGreaterEqual(len(res["features"]), 15)

        # Check percentages sum to approx 100%
        pct_sum = sum(f["percentage"] for f in res["features"])
        self.assertAlmostEqual(pct_sum, 100.0, delta=1.0)

        # Top features must be sorted descending
        for i in range(len(res["features"]) - 1):
            self.assertGreaterEqual(res["features"][i]["importance"], res["features"][i + 1]["importance"])

    def test_local_tree_path_decomposition_mathematical_identity(self):
        from backend.explainability import explain_individual_prediction
        payload = {
            "crop": "Wheat",
            "soil_type": "Loamy",
            "nitrogen": 45.0,
            "phosphorus": 40.0,
            "potassium": 35.0,
            "soil_ph": 6.8,
            "organic_carbon": 0.55
        }
        res = explain_individual_prediction(payload)
        self.assertTrue(res["success"])
        self.assertIn("predicted_fertilizer", res)
        self.assertGreater(res["confidence"], 0.0)
        self.assertGreater(res["base_probability"], 0.0)
        self.assertIn("narrative", res)
        self.assertIn("disclaimer", res)

        # Mathematical decomposition check: Base + Sum(Contributions) ≈ Confidence
        contrib_sum = sum(c["impact_percent"] for c in res["all_feature_contributions"])
        reconstructed_confidence = res["base_probability"] + contrib_sum
        self.assertAlmostEqual(reconstructed_confidence, res["confidence"], delta=0.5)

    def test_api_explain_global_endpoint(self):
        res = self.client.get("/api/explain/global")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertIn("features", data)
        self.assertGreaterEqual(len(data["features"]), 10)

    def test_api_explain_local_endpoint(self):
        payload = {
            "crop": "Rice",
            "soil_type": "Clay",
            "nitrogen": 60.0,
            "phosphorus": 25.0,
            "potassium": 50.0,
            "soil_ph": 5.8
        }
        res = self.client.post("/api/explain/local", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertIn("predicted_fertilizer", data)
        self.assertIn("positive_drivers", data)
        self.assertIn("narrative", data)

    def test_explainability_page_route(self):
        res = self.client.get("/explainability")
        self.assertEqual(res.status_code, 200)

    def test_recommend_endpoint_includes_xai_explanation(self):
        payload = {
            "crop": "Wheat",
            "soil_type": "Loamy",
            "nitrogen": 45.0,
            "phosphorus": 40.0,
            "potassium": 35.0,
            "ph": 6.8,
            "plot_area": 1.0,
            "area_unit": "Acre",
            "budget": 2000.0
        }
        res = self.client.post("/api/recommend", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertIn("xai_explanation", data)
        if data["xai_explanation"]:
            self.assertIn("positive_drivers", data["xai_explanation"])
            self.assertIn("narrative", data["xai_explanation"])

        # Cleanup
        if data.get("id"):
            delete_recommendation(data["id"])


class TestMarketplace(unittest.TestCase):
    """Verifies Marketplace Catalog, Demonstration Crop Listings, and Sample Submission."""

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_marketplace_page_route(self):
        res = self.client.get("/marketplace")
        self.assertEqual(res.status_code, 200)

    def test_marketplace_fertilizers_catalog_endpoint(self):
        res = self.client.get("/api/marketplace/fertilizers")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("is_demo"))
        self.assertIn("fertilizers", data)
        self.assertGreaterEqual(len(data["fertilizers"]), 5)

    def test_marketplace_crops_get_endpoint(self):
        res = self.client.get("/api/marketplace/crops")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertIn("listings", data)
        self.assertGreaterEqual(len(data["listings"]), 1)

    def test_marketplace_crops_post_sample_listing(self):
        payload = {
            "farmer_name": "Kuldeep Singh",
            "location": "Bathinda, Punjab",
            "crop": "Wheat",
            "quantity_quintals": 50.0,
            "expected_price": 2300.0,
            "contact": "98765-XXXXX"
        }
        res = self.client.post("/api/marketplace/crops", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertIn("listing_id", data)

    def test_marketplace_crops_post_validation(self):
        bad_payload = {
            "farmer_name": "Test",
            "quantity_quintals": -5.0,
            "expected_price": 0.0
        }
        res = self.client.post("/api/marketplace/crops", json=bad_payload)
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data.get("success"))


if __name__ == "__main__":
    unittest.main()