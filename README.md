# Smart Farming: Crop Recommendation and Yield Prediction

Machine-learning models for crop recommendation and crop-yield prediction using agro-environmental data.

## Overview

This project investigates two complementary machine-learning tasks:

1. **Crop Recommendation (Classification)** – recommending the most suitable crop based on soil and environmental conditions.
2. **Yield Prediction (Regression)** – estimating expected crop yield from environmental and management variables.

The project extends previous Smart Farming work and applies a complete machine-learning workflow including preprocessing, model training, validation, evaluation, and model interpretation.

## Team

- Aster Noel Dsouza (29211)
- David Heleno Bebiano Da Costa Morais (29400)

## Project Category

Tabular Data — Multi-Class Classification and Regression

## Datasets

### Crop Recommendation Dataset

Features:
- Nitrogen
- Phosphorus
- Potassium
- Temperature
- Humidity
- pH
- Rainfall

Target:
- Recommended Crop

### Agriculture Crop Yield Dataset

Features:
- Region
- Soil Type
- Crop Type
- Rainfall
- Temperature
- Fertilizer Usage
- Irrigation Usage
- Weather Conditions
- Days to Harvest

Target:
- Yield (tons per hectare)

### Supplementary Dataset

- OECD Agricultural Output – Crop Production Statistics

## Methods

### Classification

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

### Regression

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

### Evaluation

Classification:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Regression:
- R²
- RMSE
- MAE

### Explainability

- Permutation Importance
- SHAP

### Optional Extension

Climate-scenario sensitivity analysis to assess model robustness under changes in temperature and rainfall.

## Repository Structure

```text
data/
notebooks/
src/
outputs/
report/
project_proposal.md
README.md
```

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Run notebooks or project scripts according to the project workflow.

## AI Usage

Any AI-generated code included in the final project will be documented according to course requirements, including prompts used and subsequent manual modifications.

## References

- Raschka, S., Liu, Y., & Mirjalili, V. (2022). *Machine Learning with PyTorch and Scikit-Learn*. Packt.
- Crop Recommendation Dataset (Kaggle).
- Agriculture Crop Yield Dataset (Kaggle).
- OECD Agricultural Output – Crop Production Statistics.
