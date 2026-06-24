---
title: Smart Farming Crop & Yield
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Smart Farming — Crop Recommendation & Yield Prediction

Gradio app for the GreenDS *Practical Machine Learning* final project.

- **Crop recommender (Track A):** soil + climate → top-3 crops, with a +2 °C
  climate-stability check on the top pick.
- **Yield estimator (Track B):** region / soil / crop / management → yield (t/ha).

## Deploy to Hugging Face Spaces

1. Train the models locally so the artifacts exist:
   ```bash
   python run_pipeline.py          # creates outputs/models/*.joblib
   ```
2. Copy the two artifacts next to the app:
   ```bash
   mkdir -p app/models
   cp outputs/models/track_A_classifier.joblib app/models/
   cp outputs/models/track_B_regressor.joblib app/models/
   ```
3. Create a new **Gradio** Space on Hugging Face and upload the contents of
   `app/` (`app.py`, `requirements.txt`, `README.md`, `models/`).

## Run locally

```bash
pip install -r requirements.txt
python app.py        # opens http://127.0.0.1:7860
```
