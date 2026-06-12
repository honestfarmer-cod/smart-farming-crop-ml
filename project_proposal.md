# Project Proposal

**Project title:** Explainable Machine Learning for Crop Recommendation and Yield Prediction

**Project category:** Tabular data — multi-class classification + regression

**Course:** Practical Machine Learning (Applied Machine Learning), GreenDS MSc 2025/2026

**Team members:**
- Aster Noel Dsouza — Student ID 29211
- David Heleno Bebiano Da Costa Morais — Student ID 29400

**Repository:** _[(https://github.com/honestfarmer-cod/ML-Smart-Farming-Crop-Yield-Prediction-Recommendation-System)]_

---

## Project plan

### Problem statement

Smart-farming decision support involves two closely related questions: which crop should be planted under given soil and environmental conditions, and what yield can be expected from that decision. This project addresses both tasks using machine learning. The first task is crop recommendation (classification), while the second is yield prediction (regression). Beyond predictive performance, the project will investigate model interpretability and robustness under changing environmental conditions.

The project extends a previous Smart Farming prototype developed during an earlier course and re-implements it using a structured machine-learning workflow aligned with the Practical Machine Learning curriculum.

### Dataset

Track A uses the Crop Recommendation dataset containing 2,200 observations across 22 crop classes with seven environmental and soil features: nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall.

Track B uses the Agriculture Crop Yield dataset containing approximately one million observations with information on region, soil type, crop type, weather conditions, irrigation, fertilizer use, and yield.

As a supplementary analysis, OECD crop production data may be used to compare broader trends and support visualization and interpretation.

### Method

A complete scikit-learn pipeline will be developed, including preprocessing, feature transformation, model training, cross-validation, and hyperparameter tuning.

Classification models will include:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Regression models will include:
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

Model interpretation will be explored using permutation importance and SHAP.

As an extension, climate-scenario sensitivity analyses may be performed by modifying temperature and rainfall variables to assess model robustness under changing environmental conditions.

### Challenges

Key challenges include selecting appropriate validation strategies, avoiding data leakage, handling large datasets efficiently, interpreting model behaviour, and assessing how well model conclusions generalize beyond the training data.

### Evaluation

Classification models will be evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

Regression models will be evaluated using:
- R²
- RMSE
- MAE

Model performance will be compared using cross-validation results and statistical comparisons where appropriate. Additional analyses may examine model sensitivity under different environmental conditions.

---

## Planned Use of Course Concepts

This project is intended to apply several topics covered in the Practical Machine Learning course. We plan to use data preprocessing and machine-learning pipelines to prepare the datasets, followed by classification and regression models as baseline approaches. More advanced methods such as decision trees, random forests, and XGBoost will be explored and compared. Model selection will be supported through cross-validation and hyperparameter tuning, while performance will be assessed using appropriate evaluation metrics for both classification and regression tasks. We also plan to investigate model interpretability techniques to better understand the factors influencing crop recommendations and yield predictions.

## Potential Connection with AVCAD

We are considering using the same overall project theme and some of the same datasets in the Analysis and Visualization of Complex Agro-Environmental Data (AVCAD) course. If feasible, the AVCAD component would focus on exploratory data analysis, statistical analysis, clustering, dimensionality reduction, and visualisation of agro-environmental patterns within the data. Although the datasets and domain may overlap, the analyses, objectives, and final deliverables for the two courses will be developed independently to satisfy the requirements of each course.

## References

- Raschka, S., Liu, Y., & Mirjalili, V. (2022). *Machine Learning with PyTorch and Scikit-Learn*. Packt.
- Ingle, A. *Crop Recommendation Dataset*. Kaggle.
- Otieno, S. *Agriculture Crop Yield Dataset*. Kaggle.
- OECD Agricultural Output – Crop Production Statistics.
