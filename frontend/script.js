/**
 * AGRISMART - FERTILIZER OPTIMIZATION RECOMMENDER JAVASCRIPT
 * High-performance, modular agricultural frontend controller.
 */

const API_BASE = "";
let nutrientChart = null;

// ============================================================
// INITIALIZATION
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
    initAuthStatus();
    loadPresets();
    loadHistory();
    loadDatasetStats();
    setupEventListeners();
});

// ============================================================
// EVENT LISTENERS & SETUP
// ============================================================
function setupEventListeners() {
    const mainForm = document.getElementById("recommendForm") || document.getElementById("recommendationForm");
    if (mainForm) {
        mainForm.addEventListener("submit", handleRecommendationSubmit);
    }

    const quickForm = document.getElementById("quickRecommendForm");
    if (quickForm) {
        quickForm.addEventListener("submit", handleRecommendationSubmit);
    }
}

// ============================================================
// PRESETS & QUICK AUTO-FILL
// ============================================================
async function loadPresets() {
    const container = document.getElementById("presetChips");
    if (!container) return;

    try {
        const res = await fetch(`${API_BASE}/api/presets`);
        const data = await res.json();

        if (data.success && data.presets) {
            container.innerHTML = "";
            data.presets.forEach(p => {
                const btn = document.createElement("button");
                btn.type = "button";
                btn.className = "preset-chip";
                btn.innerHTML = `🌾 <strong>${p.crop}</strong> (${p.tag})`;
                btn.onclick = () => applyPreset(p);
                container.appendChild(btn);
            });
        }
    } catch (e) {
        console.warn("Could not load presets:", e);
    }
}

function applyPreset(preset) {
    const setVal = (id, val) => {
        const el = document.getElementById(id);
        if (el && val !== undefined) el.value = val;
    };

    setVal("crop", preset.crop);
    setVal("soilType", preset.soil_type);
    setVal("soil_type", preset.soil_type);
    setVal("nitrogen", preset.nitrogen);
    setVal("phosphorus", preset.phosphorus);
    setVal("potassium", preset.potassium);
    setVal("ph", preset.soil_ph);
    setVal("soil_ph", preset.soil_ph);
    setVal("moisture", preset.moisture);
    setVal("organicCarbon", preset.organic_carbon);
    setVal("organic_carbon", preset.organic_carbon);
    setVal("ec", preset.electrical_conductivity);
    setVal("electrical_conductivity", preset.electrical_conductivity);
    setVal("temperature", preset.temperature);
    setVal("humidity", preset.humidity);
    setVal("rainfall", preset.rainfall);
    setVal("growthStage", preset.growth_stage);
    setVal("growth_stage", preset.growth_stage);
    setVal("season", preset.season);
    setVal("irrigationType", preset.irrigation_type);
    setVal("irrigation_type", preset.irrigation_type);
    setVal("region", preset.region);
    setVal("plotArea", preset.plot_area);
    setVal("plot_area", preset.plot_area);
    setVal("areaUnit", preset.area_unit);
    setVal("budget", preset.budget);

    showToast(`Loaded Preset: ${preset.name}`, "info");

    const submitBtn = document.getElementById("recommendBtn") || document.getElementById("generateButton");
    if (submitBtn) {
        submitBtn.scrollIntoView({ behavior: "smooth", block: "center" });
    }
}

// ============================================================
// RECOMMENDATION FORM SUBMISSION
// ============================================================
async function handleRecommendationSubmit(e) {
    e.preventDefault();

    const submitBtn = document.getElementById("recommendBtn") || document.getElementById("generateButton");
    const originalText = submitBtn ? submitBtn.innerHTML : "Analyze";
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = "⏳ Processing Soil Data & Optimizing...";
    }

    const getNum = (id, def = 0) => {
        const el = document.getElementById(id);
        return el ? parseFloat(el.value) || def : def;
    };
    const getStr = (id, def = "") => {
        const el = document.getElementById(id);
        return el ? el.value.trim() : def;
    };

    const payload = {
        crop: getStr("crop") || getStr("cropType", "Wheat"),
        soil_type: getStr("soilType") || getStr("soil_type", "Loamy"),
        nitrogen: getNum("nitrogen", 50),
        phosphorus: getNum("phosphorus", 50),
        potassium: getNum("potassium", 50),
        soil_ph: getNum("ph") || getNum("soil_ph", 6.5),
        moisture: getNum("moisture", 30),
        organic_carbon: getNum("organicCarbon") || getNum("organic_carbon", 0.5),
        electrical_conductivity: getNum("ec") || getNum("electrical_conductivity", 1.0),
        temperature: getNum("temperature", 25),
        humidity: getNum("humidity", 60),
        rainfall: getNum("rainfall", 1000),
        growth_stage: getStr("growthStage") || getStr("growth_stage", "Vegetative"),
        season: getStr("season", "Kharif"),
        irrigation_type: getStr("irrigationType") || getStr("irrigation_type", "Canal"),
        region: getStr("region", "North"),
        plot_area: getNum("plotArea") || getNum("plot_area", 1.0),
        area_unit: getStr("areaUnit", "Acre"),
        budget: getNum("budget", 2000)
    };

    try {
        const response = await fetch(`${API_BASE}/api/recommend`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.message || "Failed to compute fertilizer optimization.");
        }

        renderRecommendationResult(data, payload);
        loadHistory();
        showToast("Fertilizer Recommendation & Cost Plan Generated!", "success");

        // Save last recommendation in localStorage for Soil Card generator
        localStorage.setItem("last_fertilizer_rec", JSON.stringify({ ...data, inputs: payload }));

    } catch (err) {
        console.error("Submission Error:", err);
        showToast(err.message, "error");
    } finally {
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalText;
        }
    }
}

// ============================================================
// RENDER RECOMMENDATION RESULT
// ============================================================
function renderRecommendationResult(data, inputs) {
    const resultCard = document.getElementById("resultCard") || document.getElementById("result");
    if (!resultCard) return;

    resultCard.classList.add("active");
    resultCard.style.display = "block";

    // Set text elements
    const setText = (id, text) => {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    };

    setText("fertilizerName", data.fertilizer || data.recommended_fertilizer);
    setText("fertilizerNPK", `NPK: ${data.fertilizer_npk || 'Balanced'}`);
    setText("dosageKg", `${data.dosage_kg} kg`);
    setText("bagsCount", `${data.bags_count} Bags (${data.bag_size_kg}kg/bag)`);
    setText("estimatedCost", `₹ ${data.estimated_cost.toLocaleString('en-IN')}`);
    setText("savingsAmount", `₹ ${data.savings_amount.toLocaleString('en-IN')}`);
    setText("standardCost", `₹ ${data.standard_cost.toLocaleString('en-IN')}`);
    setText("fertDesc", data.description);
    setText("appMethod", `Application Method: ${data.application_method}`);

    // Soil alerts
    const alertsContainer = document.getElementById("soilAlertsList");
    if (alertsContainer && data.soil_alerts) {
        alertsContainer.innerHTML = data.soil_alerts.map(a => `<li>${a}</li>`).join("");
    }

    // Application schedule
    const scheduleContainer = document.getElementById("scheduleList");
    if (scheduleContainer && data.schedule) {
        scheduleContainer.innerHTML = data.schedule.map(s => `
            <div class="schedule-item">
                <h4>${s.stage} — <strong>${s.quantity_kg} kg (${s.portion})</strong></h4>
                <p>${s.instructions}</p>
            </div>
        `).join("");
    }

    // Render XAI attribution snippet if available
    const xaiBox = document.getElementById("xaiResultSnippet");
    const xaiNarrative = document.getElementById("xaiNarrativeText");
    const xaiDrivers = document.getElementById("xaiTopDriversTags");
    if (xaiBox && data.xai_explanation) {
        xaiBox.style.display = "block";
        if (xaiNarrative && data.xai_explanation.narrative) {
            xaiNarrative.innerHTML = data.xai_explanation.narrative.replace(/\*\*(.*?)\*\*/g, '<b>$1</b>');
        }
        if (xaiDrivers && data.xai_explanation.positive_drivers) {
            xaiDrivers.innerHTML = data.xai_explanation.positive_drivers.slice(0, 4).map(d => 
                `<span style="background:rgba(255,255,255,0.25); padding:3px 8px; border-radius:12px; font-size:11px; color:#fff;">+${d.percentage_impact.toFixed(1)}% ${d.display_name}</span>`
            ).join("");
        }
    }

    // Render Chart.js
    renderNutrientChart(data);

    // Scroll smoothly to result
    resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ============================================================
// CHART.JS VISUALIZATION
// ============================================================
function renderNutrientChart(data) {
    const canvas = document.getElementById("nutrientChart");
    if (!canvas || typeof Chart === "undefined") return;

    if (nutrientChart) {
        nutrientChart.destroy();
    }

    const ctx = canvas.getContext("2d");
    const labels = ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"];
    const available = [
        data.soil_available?.N || 50,
        data.soil_available?.P || 40,
        data.soil_available?.K || 45
    ];
    const required = [
        data.crop_requirements?.N || 120,
        data.crop_requirements?.P || 60,
        data.crop_requirements?.K || 50
    ];

    nutrientChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Available in Soil (kg/plot)",
                    data: available,
                    backgroundColor: "rgba(34, 160, 107, 0.7)",
                    borderColor: "#22a06b",
                    borderWidth: 1.5,
                    borderRadius: 6
                },
                {
                    label: "Crop Requirement (kg/plot)",
                    data: required,
                    backgroundColor: "rgba(230, 126, 34, 0.7)",
                    borderColor: "#e67e22",
                    borderWidth: 1.5,
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: "top", labels: { font: { size: 12, weight: "bold" } } },
                title: { display: true, text: "Soil Nutrient Status vs Crop Demand", font: { size: 14 } }
            },
            scales: {
                y: { beginAtZero: true, title: { display: true, text: "Nutrient Amount (kg)" } }
            }
        }
    });
}

// ============================================================
// HISTORY MANAGER
// ============================================================
async function loadHistory() {
    const container = document.getElementById("historyList") || document.getElementById("historyTableBody");
    if (!container) return;

    try {
        const res = await fetch(`${API_BASE}/api/history`);
        const data = await res.json();

        if (data.success && data.history) {
            if (data.history.length === 0) {
                container.innerHTML = `<div class="card-desc">No past recommendations yet. Generate your first recommendation above.</div>`;
                return;
            }

            container.innerHTML = "";
            data.history.forEach(rec => {
                const item = document.createElement("div");
                item.className = "history-item";
                item.style.cssText = "background:#fff; border:1px solid #dce7dc; padding:16px; border-radius:10px; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center;";
                item.innerHTML = `
                    <div>
                        <strong style="color:#176b3a; font-size:16px;">🌱 ${escapeHtml(rec.recommended_fertilizer)}</strong>
                        <div style="color:#5e7364; font-size:13px; margin-top:3px;">
                            Crop: <b>${escapeHtml(rec.crop)}</b> | Area: <b>${rec.plot_area || 1} ${rec.area_unit || 'Acre'}</b> | Dosage: <b>${rec.dosage_kg || '--'} kg</b>
                        </div>
                        <small style="color:#888;">Date: ${new Date(rec.created_at).toLocaleString()}</small>
                    </div>
                    <div style="display:flex; gap:8px;">
                        <button class="btn btn-primary" style="padding:6px 12px; font-size:12px;" onclick='viewSoilCard(${JSON.stringify(rec)})'>📄 Card</button>
                        <button class="btn btn-outline" style="padding:6px 12px; font-size:12px; color:#d93025; border-color:#d93025;" onclick="deleteHistory(${rec.id})">🗑️</button>
                    </div>
                `;
                container.appendChild(item);
            });
        }
    } catch (e) {
        console.warn("Could not load history:", e);
    }
}

async function deleteHistory(id) {
    if (!confirm("Are you sure you want to delete this recommendation record?")) return;
    try {
        const res = await fetch(`${API_BASE}/api/history/${id}`, { method: "DELETE" });
        const data = await res.json();
        if (data.success) {
            showToast("Record removed.", "info");
            loadHistory();
        }
    } catch (e) {
        showToast("Error deleting record.", "error");
    }
}

function viewSoilCard(rec) {
    localStorage.setItem("last_fertilizer_rec", JSON.stringify(rec));
    window.location.href = "/soil-card";
}

// ============================================================
// DATASET STATS
// ============================================================
async function loadDatasetStats() {
    try {
        const res = await fetch(`${API_BASE}/api/statistics`);
        const data = await res.json();
        if (data.success) {
            const setText = (id, val) => {
                const el = document.getElementById(id);
                if (el && val !== undefined) el.textContent = val;
            };
            setText("statTotalRecs", data.total_recommendations || 0);
            setText("statTotalSavings", `₹ ${(data.averages?.total_savings || 0).toLocaleString('en-IN')}`);
        }
    } catch (e) {
        // silent fallback
    }
}

// ============================================================
// AUTHENTICATION
// ============================================================
async function initAuthStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/me`);
        const data = await res.json();
        const userDisplay = document.getElementById("userNameDisplay");
        if (userDisplay && data.user) {
            userDisplay.textContent = data.user.name || "Farmer";
        }
    } catch (e) {}
}

async function logout() {
    await fetch(`${API_BASE}/api/logout`, { method: "POST" });
    showToast("Logged out.", "info");
    setTimeout(() => window.location.href = "/", 600);
}

// ============================================================
// UTILITIES
// ============================================================
function showToast(msg, type = "info") {
    let toast = document.getElementById("toastNotification");
    if (!toast) {
        toast = document.createElement("div");
        toast.id = "toastNotification";
        toast.className = "toast";
        document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.style.background = type === "error" ? "#d93025" : (type === "success" ? "#176b3a" : "#193121");
    toast.classList.add("show");
    setTimeout(() => toast.classList.remove("show"), 3500);
}

function escapeHtml(str) {
    if (!str) return "";
    return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}