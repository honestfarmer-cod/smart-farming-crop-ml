# Use of AI: this file was drafted with an AI assistant and then reviewed
# and adapted by the team for this project.
# prompt: "create a configuration module with the dataset paths, a fixed random seed, the feature lists for both tracks, and the temperature and rainfall scenarios"
# Modifications: <describe the changes you made, e.g. parameter values, fixes>

"""
Configuration values shared across the project: dataset paths, the random seed
used everywhere for reproducibility, the feature lists for both tracks, and the
temperature and rainfall scenarios used in the climate sensitivity analysis.
"""
from pathlib import Path

# --- folders -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
METRICS = OUTPUTS / "metrics"
MODELS = OUTPUTS / "models"
for _d in (FIGURES, METRICS, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

# --- reproducibility ---------------------------------------------------------
SEED = 42

# --- Track A: crop recommendation (classification) ---------------------------
CROP_CSV = DATA / "Crop_recommendation.csv"
A_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
A_TARGET = "label"
# nicer axis labels for plots / the app
A_PRETTY = {
    "N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)",
    "temperature": "Temperature (°C)", "humidity": "Humidity (%)",
    "ph": "Soil pH", "rainfall": "Rainfall (mm)",
}

# --- Track B: crop yield (regression) ----------------------------------------
# The loader prefers the full 1M-row file if it is present, otherwise it falls
# back to the 60k stratified sample that ships in the repo.
YIELD_FULL_CSV = DATA / "crop_yield.csv"
YIELD_SAMPLE_CSV = DATA / "crop_yield_sample.csv"
B_NUM_FEATURES = ["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest"]
B_BOOL_FEATURES = ["Fertilizer_Used", "Irrigation_Used"]
B_CAT_FEATURES = ["Region", "Soil_Type", "Crop", "Weather_Condition"]
B_TARGET = "Yield_tons_per_hectare"
# Track B can be huge; cap rows used for modelling so it runs on a laptop CPU.
B_MODEL_ROWS = 80_000

# --- climate-scenario stress test (the novel part) ---------------------------
# Each scenario is (label, delta_temperature_°C, rainfall_multiplier).
CLIMATE_SCENARIOS = [
    ("baseline",        0.0, 1.00),
    ("+1.5°C",          1.5, 1.00),
    ("+2°C",            2.0, 1.00),
    ("+3°C",            3.0, 1.00),
    ("-10% rain",       0.0, 0.90),
    ("-20% rain",       0.0, 0.80),
    ("-30% rain",       0.0, 0.70),
    ("+10% rain",       0.0, 1.10),
    ("+2°C, -20% rain", 2.0, 0.80),   # a plausible combined drought-warming case
]

# --- plotting ----------------------------------------------------------------
FIG_DPI = 120


def savefig(fig, name):
    """Save a figure to outputs/figures and report the path."""
    path = FIGURES / name
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    return path
