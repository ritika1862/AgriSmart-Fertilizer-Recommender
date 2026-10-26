"""
AgriSmart - Fertilizer Cost Optimizer Engine
Formulates and solves the nutrient cost minimization problem using verified
ICAR crop nutrient requirements, chemical stoichiometric content, and market pricing benchmarks.
"""

import math
from typing import Dict, Any, List, Optional
import numpy as np
from scipy.optimize import linprog

# ============================================================
# VERIFIED FERTILIZER CATALOG & PRICING TRANSPARENCY METADATA
# ============================================================
# Price classifications:
# - "Statutory Subsidized MRP": Legally fixed Maximum Retail Price under Fertilizer Control Order (FCO).
# - "Official Indicative Subsidy Benchmark": Standard rate set by major cooperative manufacturers (IFFCO/KRIBHCO) under Nutrient Based Subsidy (NBS).
# - "Indicative Market Benchmark": Prevailing retail market estimates which may fluctuate regionally.

FERTILIZER_CATALOG: Dict[str, Dict[str, Any]] = {
    "Urea": {
        "n_percent": 46.0,
        "p_percent": 0.0,
        "k_percent": 0.0,
        "bag_size_kg": 45,
        "price_per_bag": 266.50,
        "price_type": "Statutory Subsidized MRP",
        "price_note": "Statutory MRP fixed by Dept. of Fertilizers, Govt. of India for 45 kg subsidized bag.",
        "primary_nutrient": "Nitrogen",
        "description": "High-analysis nitrogenous fertilizer providing quick-release ammoniacal/amide nitrogen."
    },
    "DAP": {
        "n_percent": 18.0,
        "p_percent": 46.0,
        "k_percent": 0.0,
        "bag_size_kg": 50,
        "price_per_bag": 1350.00,
        "price_type": "Official Indicative Subsidy Benchmark",
        "price_note": "Governed under NBS scheme; ₹1,350/50kg is the uniform benchmark maintained by IFFCO/cooperatives.",
        "primary_nutrient": "Phosphorus + Starter Nitrogen",
        "description": "Di-Ammonium Phosphate supplies water-soluble phosphate for root proliferation along with starter nitrogen."
    },
    "MOP": {
        "n_percent": 0.0,
        "p_percent": 0.0,
        "k_percent": 60.0,
        "bag_size_kg": 50,
        "price_per_bag": 1700.00,
        "price_type": "Indicative Market Benchmark",
        "price_note": "Import-dependent potash price. Prevailing market range ₹1,650 - ₹1,750 per 50 kg bag.",
        "primary_nutrient": "Potassium",
        "description": "Muriate of Potash (Potassium Chloride) builds disease tolerance, drought resistance, and grain firmness."
    },
    "NPK_10_26_26": {
        "n_percent": 10.0,
        "p_percent": 26.0,
        "k_percent": 26.0,
        "bag_size_kg": 50,
        "price_per_bag": 1470.00,
        "price_type": "Indicative Market Benchmark",
        "price_note": "Complex grade NPK benchmark for balanced secondary application.",
        "primary_nutrient": "Balanced Complex (N-P-K)",
        "description": "Complex granulated fertilizer providing homogenous proportion of phosphorus and potassium with starter N."
    },
    "SSP": {
        "n_percent": 0.0,
        "p_percent": 16.0,
        "k_percent": 0.0,
        "bag_size_kg": 50,
        "price_per_bag": 550.00,
        "price_type": "Indicative Market Benchmark",
        "price_note": "Economical indigenous source of phosphate + 11% Sulphur. Market range ₹500 - ₹600 per 50 kg bag.",
        "primary_nutrient": "Phosphorus + 11% Sulphur",
        "description": "Single Super Phosphate delivers readily available water-soluble phosphate with essential sulphur."
    }
}

# ============================================================
# VERIFIED ICAR RECOMMENDED DOSES OF FERTILIZERS (RDF)
# Units: kg active nutrient per hectare (and scaled for farm acre calculations)
# ============================================================
CROP_NUTRIENT_DEMANDS: Dict[str, Dict[str, Any]] = {
    "Wheat": {
        "target_n": 120.0, "target_p": 60.0, "target_k": 40.0,
        "ideal_ph_min": 6.0, "ideal_ph_max": 7.5,
        "source": "ICAR-Indian Institute of Wheat and Barley Research (IIWBR) standard RDF"
    },
    "Rice": {
        "target_n": 100.0, "target_p": 50.0, "target_k": 50.0,
        "ideal_ph_min": 5.5, "ideal_ph_max": 7.0,
        "source": "ICAR-National Rice Research Institute (NRRI) benchmark RDF"
    },
    "Maize": {
        "target_n": 120.0, "target_p": 60.0, "target_k": 50.0,
        "ideal_ph_min": 5.8, "ideal_ph_max": 7.2,
        "source": "ICAR-Indian Institute of Maize Research (IIMR) benchmark RDF"
    },
    "Cotton": {
        "target_n": 100.0, "target_p": 50.0, "target_k": 50.0,
        "ideal_ph_min": 6.0, "ideal_ph_max": 8.0,
        "source": "ICAR-Central Institute for Cotton Research (CICR) hybrid RDF"
    },
    "Sugarcane": {
        "target_n": 150.0, "target_p": 75.0, "target_k": 75.0,
        "ideal_ph_min": 6.0, "ideal_ph_max": 7.5,
        "source": "ICAR-Indian Institute of Sugarcane Research (IISR) recommendation"
    },
    "Potato": {
        "target_n": 120.0, "target_p": 80.0, "target_k": 100.0,
        "ideal_ph_min": 5.0, "ideal_ph_max": 6.5,
        "source": "ICAR-Central Potato Research Institute (CPRI) tuber requirement"
    },
    "Tomato": {
        "target_n": 100.0, "target_p": 60.0, "target_k": 60.0,
        "ideal_ph_min": 6.0, "ideal_ph_max": 7.0,
        "source": "ICAR-Indian Institute of Horticultural Research (IIHR) standard"
    },
    "Soybean": {
        "target_n": 30.0, "target_p": 60.0, "target_k": 40.0,
        "ideal_ph_min": 6.0, "ideal_ph_max": 7.0,
        "source": "ICAR-Indian Institute of Soybean Research (IISR) symbiotic RDF"
    },
    "Groundnut": {
        "target_n": 25.0, "target_p": 50.0, "target_k": 40.0,
        "ideal_ph_min": 5.8, "ideal_ph_max": 6.8,
        "source": "ICAR-Directorate of Groundnut Research (DGR) starter RDF"
    }
}


def get_fertilizer_catalog_list(custom_prices: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """Returns the full catalog of available fertilizers with price transparency tags."""
    catalog = []
    prices = custom_prices or {}
    for key, item in FERTILIZER_CATALOG.items():
        price = float(prices.get(key, item["price_per_bag"]))
        catalog.append({
            "key": key,
            "name": key.replace("_", " "),
            "n_percent": item["n_percent"],
            "p_percent": item["p_percent"],
            "k_percent": item["k_percent"],
            "bag_size_kg": item["bag_size_kg"],
            "price_per_bag": price,
            "price_type": item["price_type"],
            "price_note": item["price_note"],
            "primary_nutrient": item["primary_nutrient"],
            "description": item["description"]
        })
    return catalog


def validate_optimizer_inputs(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates user input parameters. Raises ValueError with a plain,
    actionable explanation if essential data is missing or invalid.
    """
    crop = str(data.get("crop") or "").strip()
    if not crop:
        raise ValueError("Crop name is required. Please select or enter a crop.")

    if crop not in CROP_NUTRIENT_DEMANDS:
        valid_crops = ", ".join(sorted(CROP_NUTRIENT_DEMANDS.keys()))
        raise ValueError(
            f"Verified ICAR nutrient standards are currently not available for '{crop}'. "
            f"Supported crops with reliable data are: {valid_crops}. "
            "Please select a supported crop to ensure recommendations are scientifically grounded."
        )

    try:
        plot_area = float(data.get("plot_area", 1.0))
    except (TypeError, ValueError):
        raise ValueError("Invalid plot area. Please enter a valid numeric value.")
    if plot_area <= 0:
        raise ValueError("Farm plot area must be greater than zero.")

    area_unit = str(data.get("area_unit") or "Acre").strip()
    if area_unit.lower() not in ("acre", "hectare"):
        area_unit = "Acre"

    try:
        budget = float(data.get("budget", 3000.0))
    except (TypeError, ValueError):
        raise ValueError("Invalid budget value. Please enter a valid numeric value.")
    if budget < 0:
        raise ValueError("Budget cannot be negative.")

    # Soil nutrient measurements (0 - 300 range standard for Indian soil testing)
    def parse_nutrient(val, name):
        try:
            num = float(val) if val is not None else 50.0
        except (TypeError, ValueError):
            raise ValueError(f"Invalid value for {name}. Please provide a numeric soil test value.")
        if num < 0:
            raise ValueError(f"{name} level cannot be negative.")
        return num

    nitrogen = parse_nutrient(data.get("nitrogen"), "Nitrogen (N)")
    phosphorus = parse_nutrient(data.get("phosphorus"), "Phosphorus (P)")
    potassium = parse_nutrient(data.get("potassium"), "Potassium (K)")

    return {
        "crop": crop,
        "plot_area": plot_area,
        "area_unit": area_unit,
        "budget": budget,
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium,
        "custom_prices": data.get("custom_prices") or {}
    }


def optimize_fertilizer_cost(
    data: Any = None,
    plot_area: Optional[float] = None,
    area_unit: Optional[str] = None,
    budget: Optional[float] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Executes cost optimization across verified agricultural strategies.
    Accepts either:
    1. A single dictionary parameter `data`, or
    2. Positional / keyword arguments: `crop, plot_area, area_unit, budget`.
    Ensures:
    1. 100% of defined N, P, K deficits are satisfied.
    2. Uses real fertilizer nutrient fractions and verified pricing.
    3. Clearly flags budget shortfalls and price transparency types.
    """
    if isinstance(data, dict):
        payload = dict(data)
    elif isinstance(data, str):
        payload = {
            "crop": data,
            "plot_area": plot_area if plot_area is not None else 1.0,
            "area_unit": area_unit if area_unit is not None else "Acre",
            "budget": budget if budget is not None else 3000.0,
            **kwargs
        }
    elif data is None and "crop" in kwargs:
        payload = kwargs
        if plot_area is not None: payload["plot_area"] = plot_area
        if area_unit is not None: payload["area_unit"] = area_unit
        if budget is not None: payload["budget"] = budget
    else:
        payload = {"crop": str(data or "Wheat")}

    validated = validate_optimizer_inputs(payload)
    crop = validated["crop"]
    plot_area = validated["plot_area"]
    area_unit = validated["area_unit"]
    budget = validated["budget"]
    nitrogen = validated["nitrogen"]
    phosphorus = validated["phosphorus"]
    potassium = validated["potassium"]
    custom_prices = validated["custom_prices"]

    # Conversion to effective area in acres
    area_in_acres = plot_area if area_unit.lower() == "acre" else plot_area * 2.47105

    # Crop targets
    crop_info = CROP_NUTRIENT_DEMANDS[crop]
    target_n = crop_info["target_n"] * area_in_acres
    target_p = crop_info["target_p"] * area_in_acres
    target_k = crop_info["target_k"] * area_in_acres

    # Soil availability index
    avail_n = nitrogen * area_in_acres
    avail_p = phosphorus * area_in_acres
    avail_k = potassium * area_in_acres

    # Net nutrient deficits that must be supplied by fertilizers
    gap_n = max(0.0, target_n - avail_n)
    gap_p = max(0.0, target_p - avail_p)
    gap_k = max(0.0, target_k - avail_k)

    # Resolve active prices
    prices = {}
    for key, item in FERTILIZER_CATALOG.items():
        prices[key] = float(custom_prices.get(key, item["price_per_bag"]))

    # --------------------------------------------------------
    # STRATEGY 1: Straight Fertilizer Package (DAP + Urea + MOP)
    # The gold-standard traditional practice across Indian agriculture.
    # DAP supplies Phosphorus and starter Nitrogen. Urea tops up remaining N. MOP supplies K.
    # --------------------------------------------------------
    dap_p_per_bag = 50.0 * 0.46   # 23.0 kg P per 50kg bag
    dap_n_per_bag = 50.0 * 0.18   # 9.0 kg N per 50kg bag
    urea_n_per_bag = 45.0 * 0.46  # 20.7 kg N per 45kg bag
    mop_k_per_bag = 50.0 * 0.60   # 30.0 kg K per 50kg bag

    bags_dap = math.ceil(gap_p / dap_p_per_bag) if gap_p > 0 else 0
    n_from_dap = bags_dap * dap_n_per_bag
    p_from_dap = bags_dap * dap_p_per_bag

    rem_n_strat1 = max(0.0, gap_n - n_from_dap)
    bags_urea_strat1 = math.ceil(rem_n_strat1 / urea_n_per_bag) if rem_n_strat1 > 0 else 0
    n_from_urea_strat1 = bags_urea_strat1 * urea_n_per_bag

    bags_mop_strat1 = math.ceil(gap_k / mop_k_per_bag) if gap_k > 0 else 0
    k_from_mop_strat1 = bags_mop_strat1 * mop_k_per_bag

    cost_strat1 = round(
        (bags_dap * prices["DAP"]) +
        (bags_urea_strat1 * prices["Urea"]) +
        (bags_mop_strat1 * prices["MOP"]),
        2
    )

    strat1_items = []
    if bags_dap > 0:
        strat1_items.append({
            "fertilizer": "DAP", "bags": bags_dap, "bag_size_kg": 50,
            "total_kg": bags_dap * 50, "unit_price": prices["DAP"],
            "cost": round(bags_dap * prices["DAP"], 2),
            "n_kg": round(n_from_dap, 1), "p_kg": round(p_from_dap, 1), "k_kg": 0.0,
            "price_type": FERTILIZER_CATALOG["DAP"]["price_type"]
        })
    if bags_urea_strat1 > 0:
        strat1_items.append({
            "fertilizer": "Urea", "bags": bags_urea_strat1, "bag_size_kg": 45,
            "total_kg": bags_urea_strat1 * 45, "unit_price": prices["Urea"],
            "cost": round(bags_urea_strat1 * prices["Urea"], 2),
            "n_kg": round(n_from_urea_strat1, 1), "p_kg": 0.0, "k_kg": 0.0,
            "price_type": FERTILIZER_CATALOG["Urea"]["price_type"]
        })
    if bags_mop_strat1 > 0:
        strat1_items.append({
            "fertilizer": "MOP", "bags": bags_mop_strat1, "bag_size_kg": 50,
            "total_kg": bags_mop_strat1 * 50, "unit_price": prices["MOP"],
            "cost": round(bags_mop_strat1 * prices["MOP"], 2),
            "n_kg": 0.0, "p_kg": 0.0, "k_kg": round(k_from_mop_strat1, 1),
            "price_type": FERTILIZER_CATALOG["MOP"]["price_type"]
        })

    # --------------------------------------------------------
    # STRATEGY 2: Complex Grade Package (NPK 10:26:26 + Urea + MOP)
    # Balanced complex grade supplies N, P, and K simultaneously in granulated form.
    # --------------------------------------------------------
    npk_n_per_bag = 50.0 * 0.10   # 5.0 kg N
    npk_p_per_bag = 50.0 * 0.26   # 13.0 kg P
    npk_k_per_bag = 50.0 * 0.26   # 13.0 kg K

    bags_npk = math.ceil(gap_p / npk_p_per_bag) if gap_p > 0 else 0
    n_from_npk = bags_npk * npk_n_per_bag
    p_from_npk = bags_npk * npk_p_per_bag
    k_from_npk = bags_npk * npk_k_per_bag

    rem_n_strat2 = max(0.0, gap_n - n_from_npk)
    bags_urea_strat2 = math.ceil(rem_n_strat2 / urea_n_per_bag) if rem_n_strat2 > 0 else 0
    n_from_urea_strat2 = bags_urea_strat2 * urea_n_per_bag

    rem_k_strat2 = max(0.0, gap_k - k_from_npk)
    bags_mop_strat2 = math.ceil(rem_k_strat2 / mop_k_per_bag) if rem_k_strat2 > 0 else 0
    k_from_mop_strat2 = bags_mop_strat2 * mop_k_per_bag

    cost_strat2 = round(
        (bags_npk * prices["NPK_10_26_26"]) +
        (bags_urea_strat2 * prices["Urea"]) +
        (bags_mop_strat2 * prices["MOP"]),
        2
    )

    strat2_items = []
    if bags_npk > 0:
        strat2_items.append({
            "fertilizer": "NPK 10:26:26", "bags": bags_npk, "bag_size_kg": 50,
            "total_kg": bags_npk * 50, "unit_price": prices["NPK_10_26_26"],
            "cost": round(bags_npk * prices["NPK_10_26_26"], 2),
            "n_kg": round(n_from_npk, 1), "p_kg": round(p_from_npk, 1), "k_kg": round(k_from_npk, 1),
            "price_type": FERTILIZER_CATALOG["NPK_10_26_26"]["price_type"]
        })
    if bags_urea_strat2 > 0:
        strat2_items.append({
            "fertilizer": "Urea", "bags": bags_urea_strat2, "bag_size_kg": 45,
            "total_kg": bags_urea_strat2 * 45, "unit_price": prices["Urea"],
            "cost": round(bags_urea_strat2 * prices["Urea"], 2),
            "n_kg": round(n_from_urea_strat2, 1), "p_kg": 0.0, "k_kg": 0.0,
            "price_type": FERTILIZER_CATALOG["Urea"]["price_type"]
        })
    if bags_mop_strat2 > 0:
        strat2_items.append({
            "fertilizer": "MOP", "bags": bags_mop_strat2, "bag_size_kg": 50,
            "total_kg": bags_mop_strat2 * 50, "unit_price": prices["MOP"],
            "cost": round(bags_mop_strat2 * prices["MOP"], 2),
            "n_kg": 0.0, "p_kg": 0.0, "k_kg": round(k_from_mop_strat2, 1),
            "price_type": FERTILIZER_CATALOG["MOP"]["price_type"]
        })

    # --------------------------------------------------------
    # STRATEGY 3: Single Super Phosphate Package (SSP + Urea + MOP)
    # Recommended for oilseeds, pulses, or when phosphorus is supplemented with essential Sulphur.
    # --------------------------------------------------------
    ssp_p_per_bag = 50.0 * 0.16   # 8.0 kg P
    bags_ssp = math.ceil(gap_p / ssp_p_per_bag) if gap_p > 0 else 0
    p_from_ssp = bags_ssp * ssp_p_per_bag

    bags_urea_strat3 = math.ceil(gap_n / urea_n_per_bag) if gap_n > 0 else 0
    n_from_urea_strat3 = bags_urea_strat3 * urea_n_per_bag

    bags_mop_strat3 = math.ceil(gap_k / mop_k_per_bag) if gap_k > 0 else 0
    k_from_mop_strat3 = bags_mop_strat3 * mop_k_per_bag

    cost_strat3 = round(
        (bags_ssp * prices["SSP"]) +
        (bags_urea_strat3 * prices["Urea"]) +
        (bags_mop_strat3 * prices["MOP"]),
        2
    )

    strat3_items = []
    if bags_ssp > 0:
        strat3_items.append({
            "fertilizer": "SSP", "bags": bags_ssp, "bag_size_kg": 50,
            "total_kg": bags_ssp * 50, "unit_price": prices["SSP"],
            "cost": round(bags_ssp * prices["SSP"], 2),
            "n_kg": 0.0, "p_kg": round(p_from_ssp, 1), "k_kg": 0.0,
            "price_type": FERTILIZER_CATALOG["SSP"]["price_type"]
        })
    if bags_urea_strat3 > 0:
        strat3_items.append({
            "fertilizer": "Urea", "bags": bags_urea_strat3, "bag_size_kg": 45,
            "total_kg": bags_urea_strat3 * 45, "unit_price": prices["Urea"],
            "cost": round(bags_urea_strat3 * prices["Urea"], 2),
            "n_kg": round(n_from_urea_strat3, 1), "p_kg": 0.0, "k_kg": 0.0,
            "price_type": FERTILIZER_CATALOG["Urea"]["price_type"]
        })
    if bags_mop_strat3 > 0:
        strat3_items.append({
            "fertilizer": "MOP", "bags": bags_mop_strat3, "bag_size_kg": 50,
            "total_kg": bags_mop_strat3 * 50, "unit_price": prices["MOP"],
            "cost": round(bags_mop_strat3 * prices["MOP"], 2),
            "n_kg": 0.0, "p_kg": 0.0, "k_kg": round(k_from_mop_strat3, 1),
            "price_type": FERTILIZER_CATALOG["MOP"]["price_type"]
        })

    # --------------------------------------------------------
    # CONTINUOUS LINEAR PROGRAMMING BENCHMARK (Scipy Highs)
    # Finds theoretical absolute lower bound cost in continuous kilograms
    # --------------------------------------------------------
    # Variables: [Urea_kg, DAP_kg, MOP_kg, NPK_kg, SSP_kg]
    cost_per_kg = [
        prices["Urea"] / 45.0,
        prices["DAP"] / 50.0,
        prices["MOP"] / 50.0,
        prices["NPK_10_26_26"] / 50.0,
        prices["SSP"] / 50.0
    ]
    # Nutrient matrix (-A x <= -b)
    # N constraint: 0.46*U + 0.18*D + 0*M + 0.10*NPK + 0*SSP >= gap_n
    # P constraint: 0*U + 0.46*D + 0*M + 0.26*NPK + 0.16*SSP >= gap_p
    # K constraint: 0*U + 0*D + 0.60*M + 0.26*NPK + 0*SSP >= gap_k
    A_ub = [
        [-0.46, -0.18, 0.00, -0.10, 0.00],
        [0.00, -0.46, 0.00, -0.26, -0.16],
        [0.00, 0.00, -0.60, -0.26, 0.00]
    ]
    b_ub = [-gap_n, -gap_p, -gap_k]

    lp_res = linprog(cost_per_kg, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method="highs")
    theoretical_min_cost = round(float(lp_res.fun), 2) if lp_res.success else 0.0

    # --------------------------------------------------------
    # COMPILE & RANK STRATEGIES
    # --------------------------------------------------------
    all_strategies = [
        {
            "id": "strategy_straight",
            "name": "Straight Fertilizers (DAP + Urea + MOP)",
            "description": "Standard ICAR-recommended package. Highly available in all PACS/cooperative societies.",
            "total_cost": cost_strat1,
            "items": strat1_items,
            "supplied": {
                "n": round(n_from_dap + n_from_urea_strat1, 1),
                "p": round(p_from_dap, 1),
                "k": round(k_from_mop_strat1, 1)
            }
        },
        {
            "id": "strategy_complex",
            "name": "Complex NPK Grade (NPK 10:26:26 + Urea + MOP)",
            "description": "Granulated complex delivery for even nutrient distribution and reduced application labor.",
            "total_cost": cost_strat2,
            "items": strat2_items,
            "supplied": {
                "n": round(n_from_npk + n_from_urea_strat2, 1),
                "p": round(p_from_npk, 1),
                "k": round(k_from_npk + k_from_mop_strat2, 1)
            }
        },
        {
            "id": "strategy_ssp",
            "name": "Single Super Phosphate (SSP + Urea + MOP)",
            "description": "Economical phosphorus option supplying 11% essential Sulphur, ideal for oilseeds and pulses.",
            "total_cost": cost_strat3,
            "items": strat3_items,
            "supplied": {
                "n": round(n_from_urea_strat3, 1),
                "p": round(p_from_ssp, 1),
                "k": round(k_from_mop_strat3, 1)
            }
        }
    ]

    # Select the lowest-cost strategy that fulfills 100% of the nutrient demand
    valid_strategies = [s for s in all_strategies if s["total_cost"] > 0]
    best_strategy = min(valid_strategies, key=lambda s: s["total_cost"]) if valid_strategies else all_strategies[0]

    # Standard unguided blanket application benchmark (avg ₹1,850/acre)
    blanket_standard_cost = round(1850.0 * area_in_acres, 2)
    savings_amount = round(max(0.0, blanket_standard_cost - best_strategy["total_cost"]), 2)

    # Budget feasibility analysis
    within_budget = best_strategy["total_cost"] <= budget
    budget_deficit = round(max(0.0, best_strategy["total_cost"] - budget), 2)

    warnings = []
    if not within_budget:
        warnings.append(
            f"Budget Shortfall: Your allocated budget of ₹{budget:,.2f} is insufficient to purchase "
            f"the required fertilizer dosage (Minimum required: ₹{best_strategy['total_cost']:,.2f}). "
            f"Shortfall amount: ₹{budget_deficit:,.2f}."
        )

    if gap_n == 0 and gap_p == 0 and gap_k == 0:
        warnings.append("Soil Test Status: Soil nutrient reserves currently meet or exceed crop demand. No additional synthetic N-P-K required.")

    return {
        "success": True,
        "crop": crop,
        "crop_source": crop_info["source"],
        "plot_area": plot_area,
        "area_unit": area_unit,
        "area_acres": round(area_in_acres, 2),
        "budget": budget,
        "requirements": {
            "n": round(target_n, 1),
            "p": round(target_p, 1),
            "k": round(target_k, 1)
        },
        "soil_available": {
            "n": round(avail_n, 1),
            "p": round(avail_p, 1),
            "k": round(avail_k, 1)
        },
        "nutrient_gaps": {
            "n": round(gap_n, 1),
            "p": round(gap_p, 1),
            "k": round(gap_k, 1)
        },
        "optimal_plan": {
            "strategy_id": best_strategy["id"],
            "strategy_name": best_strategy["name"],
            "total_cost": best_strategy["total_cost"],
            "items": best_strategy["items"],
            "supplied": best_strategy["supplied"],
            "theoretical_min_cost": theoretical_min_cost,
            "within_budget": within_budget,
            "budget_deficit": budget_deficit,
            "blanket_cost": blanket_standard_cost,
            "savings_amount": savings_amount
        },
        "total_cost": best_strategy["total_cost"],
        "recommended_fertilizer": best_strategy["name"],
        "savings_amount": savings_amount,
        "within_budget": within_budget,
        "budget_status": "WITHIN_BUDGET" if within_budget else "DEFICIT",
        "budget_message": f"Allocated budget of ₹{budget:,.2f} is sufficient." if within_budget else f"Estimated package cost ₹{best_strategy['total_cost']:,.2f} exceeds your budget by ₹{budget_deficit:,.2f}.",
        "combination_details": best_strategy["items"],
        "application_schedule": [
            {"Stage": "1. Basal Application (At Sowing)", "Portion": "50% Nitrogen + 100% P & K", "Guideline": "Incorporate during seedbed preparation with adequate moisture."},
            {"Stage": "2. First Top Dressing (CRI / Tillering)", "Portion": "25% Nitrogen (Urea)", "Guideline": "Broadcast 21-25 days after sowing; irrigate within 24 hours."},
            {"Stage": "3. Second Top Dressing (Booting / Flowering)", "Portion": "25% Nitrogen (Urea)", "Guideline": "Apply before boot leaf emergence to maximize grain filling."}
        ],
        "all_strategies": all_strategies,
        "fertilizer_catalog": get_fertilizer_catalog_list(custom_prices),
        "warnings": warnings
    }
