# Use of AI: this file was drafted with an AI assistant and then reviewed
# and adapted by the team for this project.
# prompt: "build a Gradio app with two tabs - a crop recommender that shows the top crops and whether the top pick survives +2C, and a yield estimator - loading the saved scikit-learn pipelines"
# Modifications: <describe the changes you made>

"""Gradio app — Smart Farming crop recommender + yield estimator.

Two tabs:
  1. Crop recommender (Track A) — soil/climate in, top-3 crops out, plus a
     climate-stability check (does the top recommendation survive +2 °C?).
  2. Yield estimator (Track B) — region/soil/crop/management in, yield out.

The fitted pipelines are produced by ``run_pipeline.py`` (which writes them to
``outputs/models/``). Copy those two .joblib files next to this app (into an
``app/models/`` folder) before deploying to Hugging Face Spaces.
"""
import os
import numpy as np
import pandas as pd
import gradio as gr
import joblib

# --- locate the saved models (works locally and on Spaces) -------------------
SEARCH = ["models", "../outputs/models", "outputs/models"]


def _find(name):
    for d in SEARCH:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"{name} not found. Run `python run_pipeline.py` first, then copy "
        f"outputs/models/{name} into app/models/.")


clfA = joblib.load(_find("track_A_classifier.joblib"))
regB = joblib.load(_find("track_B_regressor.joblib"))
CLF, CLASSES, A_FEATURES = clfA["pipeline"], clfA["classes"], clfA["features"]
REG = regB["pipeline"]


# --- Track A: recommend -------------------------------------------------------
def recommend(N, P, K, temperature, humidity, ph, rainfall):
    x = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=A_FEATURES)
    proba = CLF.predict_proba(x)[0]
    top = {CLASSES[i]: float(proba[i]) for i in np.argsort(proba)[::-1][:3]}
    best = max(top, key=top.get)

    # climate-stability: does the top crop survive +2 C?
    xw = x.copy(); xw["temperature"] = xw["temperature"] + 2.0
    best_warm = CLASSES[int(np.argmax(CLF.predict_proba(xw)[0]))]
    if best_warm == best:
        stability = f"✅ Stable: still **{best}** at +2 °C."
    else:
        stability = f"⚠️ Fragile: flips to **{best_warm}** at +2 °C."
    return top, stability


# --- Track B: estimate yield --------------------------------------------------
def estimate_yield(Region, Soil_Type, Crop, Rainfall_mm, Temperature_Celsius,
                   Fertilizer_Used, Irrigation_Used, Weather_Condition, Days_to_Harvest):
    row = {
        "Rainfall_mm": Rainfall_mm, "Temperature_Celsius": Temperature_Celsius,
        "Days_to_Harvest": Days_to_Harvest,
        "Fertilizer_Used": int(Fertilizer_Used), "Irrigation_Used": int(Irrigation_Used),
        "rain_per_day": Rainfall_mm / max(Days_to_Harvest, 1),
        "Region": Region, "Soil_Type": Soil_Type, "Crop": Crop,
        "Weather_Condition": Weather_Condition,
    }
    y = float(REG.predict(pd.DataFrame([row]))[0])
    return f"### Estimated yield: **{y:.2f} t/ha**"


# --- UI -----------------------------------------------------------------------
with gr.Blocks(title="Smart Farming — Crop & Yield") as demo:
    gr.Markdown("# 🌱 Smart Farming — Crop Recommendation & Yield Prediction")
    gr.Markdown("Explainable, climate-aware decision support. "
                "Models trained in this repo's notebooks; see the report for details.")

    with gr.Tab("Crop recommender"):
        gr.Markdown("Enter soil nutrients and local climate to get the best crops, "
                    "with a check on whether the top pick survives +2 °C of warming.")
        with gr.Row():
            with gr.Column():
                N = gr.Slider(0, 140, 90, label="Nitrogen (N)")
                P = gr.Slider(5, 145, 42, label="Phosphorus (P)")
                K = gr.Slider(5, 205, 43, label="Potassium (K)")
                ph = gr.Slider(3.5, 10.0, 6.5, step=0.1, label="Soil pH")
            with gr.Column():
                temperature = gr.Slider(8.0, 44.0, 21.0, step=0.1, label="Temperature (°C)")
                humidity = gr.Slider(14.0, 100.0, 82.0, step=0.5, label="Humidity (%)")
                rainfall = gr.Slider(20.0, 300.0, 200.0, step=1.0, label="Rainfall (mm)")
        btn = gr.Button("Recommend", variant="primary")
        out_label = gr.Label(num_top_classes=3, label="Top crops")
        out_stab = gr.Markdown()
        btn.click(recommend, [N, P, K, temperature, humidity, ph, rainfall],
                  [out_label, out_stab])
        gr.Examples(
            [[90, 42, 43, 21, 82, 6.5, 200], [20, 130, 200, 22, 92, 5.9, 110],
             [40, 70, 80, 28, 65, 6.8, 70]],
            [N, P, K, temperature, humidity, ph, rainfall])

    with gr.Tab("Yield estimator"):
        gr.Markdown("Estimate yield for a crop under given conditions "
                    "(Track B — synthetic data, read as a methods demo).")
        with gr.Row():
            with gr.Column():
                Region = gr.Dropdown(["West", "South", "North", "East"], value="South", label="Region")
                Soil_Type = gr.Dropdown(["Sandy", "Clay", "Loam", "Silt", "Peaty", "Chalky"],
                                        value="Loam", label="Soil type")
                Crop = gr.Dropdown(["Cotton", "Rice", "Barley", "Soybean", "Wheat", "Maize"],
                                   value="Rice", label="Crop")
                Weather_Condition = gr.Dropdown(["Cloudy", "Rainy", "Sunny"], value="Rainy",
                                                label="Weather")
            with gr.Column():
                Rainfall_mm = gr.Slider(100, 1000, 600, label="Rainfall (mm)")
                Temperature_Celsius = gr.Slider(10, 45, 25, label="Temperature (°C)")
                Days_to_Harvest = gr.Slider(60, 150, 110, label="Days to harvest")
                Fertilizer_Used = gr.Checkbox(True, label="Fertilizer used")
                Irrigation_Used = gr.Checkbox(True, label="Irrigation used")
        ybtn = gr.Button("Estimate yield", variant="primary")
        yout = gr.Markdown()
        ybtn.click(estimate_yield,
                   [Region, Soil_Type, Crop, Rainfall_mm, Temperature_Celsius,
                    Fertilizer_Used, Irrigation_Used, Weather_Condition, Days_to_Harvest],
                   yout)

if __name__ == "__main__":
    demo.launch()
