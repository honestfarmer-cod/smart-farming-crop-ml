#Project Proposal – Smart Farming Crop Yield Prediction & Recommendation System (Machine Learning improved)
Project title
Smart Farming Crop Yield Prediction & Recommendation System

Project category
Tabular data (supervised regression and recommendation)

Team members
Aster Noel Dsouza – Student ID: 29211

David Heleno Bebiano Da Costa Morais – Student ID: 29400

300–500 word project description
Problem statement
We will investigate how to predict crop yield (kg/ha) from smart‑farming field trial data and soil characteristics, and how to transform these predictions into practical recommendations of the best crop variety for each soil texture class.
 This problem is interesting because small improvements in yield prediction and variety choice can have a significant impact on farmers’ income, resource efficiency, and climate‑resilient agricultural planning. 
 It also provides a realistic, structured setting to apply supervised learning, model evaluation, and basic recommendation logic to tabular data.

Dataset
The project will use field‑trial and soil datasets from our previous DMS project, already organized into raw and processed CSV files (e.g., trials_raw.csv, soil_types_cleaned.csv). 
 The main variables include agronomic management (seed rate, N–P–K fertilization, irrigation, area), categorical descriptors (crop, variety name, soil texture class), and observed yield in kg/ha. 
 To keep the project reproducible, the code also supports a “demo mode” that generates synthetic but realistic data when real CSVs are not available.

Methods / algorithms
We will implement and extend an end‑to‑end supervised regression pipeline around a RandomForestRegressor, including preprocessing (label encoding of categorical features, handling missing values), train/test split, model fitting, and prediction.
 We plan to compare the Random Forest with at least one baseline regressor (e.g., linear regression or decision tree) to highlight the trade‑offs between model complexity and performance. 
 On top of the regression model, we will implement a simple recommendation layer that, for each soil texture class, ranks available varieties by predicted yield and selects the best candidate.

Challenges
Key challenges include ensuring robust preprocessing when some CSV files are missing or partially inconsistent, handling the joint effect of multiple management variables on yield, and avoiding overfitting given the limited size of typical field‑trial datasets. 
 There is also a conceptual challenge in translating point predictions into agronomically meaningful recommendations that remain interpretable for non‑technical users.

Evaluation and analysis
Model performance will be evaluated using standard regression metrics, particularly coefficient of determination (R²) and root mean squared error (RMSE) on a held‑out test set.
 We will also analyze feature importance to understand which management and soil variables most influence predicted yield. 
 If time permits, we will perform basic statistical comparisons between models (e.g., paired error analysis) and inspect the distribution of prediction errors to assess model reliability. 
 Finally, we will qualitatively evaluate the generated variety recommendations per soil texture class and discuss limitations and possible extensions.

