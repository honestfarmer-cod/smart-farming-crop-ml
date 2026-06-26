# Smart Farming — Crop Recommendation & Yield Prediction

Explainable, climate-robust machine learning for agro-environmental decision support.

This repository contains two linked tabular machine-learning systems: a **crop recommender**
(multi-class classification) and a **crop-yield estimator** (regression). On top of the usual
training-and-accuracy workflow it adds two things that matter for real decisions — an
**explainability** layer (permutation importance + SHAP) and a **climate-scenario stress test**
that re-queries the trained models under warming and drought — plus an honest **reality-check** of
the yield data against real-world FAO figures.

- **Live app:** https://huggingface.co/spaces/asternoeld/smart-farming-crop-ml
- **Full report:** [`report/report_final.md`](report/report_final.md)
- **Original proposal:** [`project_proposal.md`](project_proposal.md)

| | |
|---|---|
| **Course** | Practical Machine Learning / Aprendizagem Automática Aplicada — GreenDS MSc 2025/2026, ISA · ULisboa |
| **Instructor** | Manuel Campagnolo |
| **Authors** | Aster Noel Dsouza (29211) · David Heleno Bebiano Da Costa Morais (29400) |
| **Category** | Tabular data — multi-class classification + regression, with explainability and a climate analysis |

---

## What the project does

**Track A — Crop recommendation (classification).** Given soil nutrients (N, P, K), pH and local
climate (temperature, humidity, rainfall), recommend the most suitable crop out of 22 options.

**Track B — Yield prediction (regression).** Given region, soil type, crop, weather and two
management flags (fertiliser, irrigation) plus rainfall, temperature and growing-season length,
estimate yield in tonnes per hectare.

**Climate-scenario stress test (the novel part).** Both trained models are re-queried on perturbed
inputs (temperature +1.5 / +2 / +3 °C and rainfall ±10–30 %, plus a combined warming+drought case)
to measure how often a recommendation *flips* (Track A) and how much predicted yield *shifts*
(Track B), then rank crops from climate-robust to climate-fragile.

**Reality check.** The yield dataset is synthetic; we compare its per-crop averages against
real-world FAO / Our World in Data yields to show where it departs from reality, rather than
presenting its numbers as agronomic truth.

---

## Datasets

| Dataset | Track | Rows | Type | Source |
|---|---|---|---|---|
| Crop Recommendation | A | 2,200 (22 classes × 100) | 7 numeric features → crop label | Kaggle (augmented from Indian climate/fertiliser records) |
| Agriculture Crop Yield | B | 1,000,000 | 4 categorical + 2 boolean + 4 numeric → yield (t/ha) | Kaggle (synthetic) |
| OECD / FAO reference yields | check | 6 crops | real-world average yields | FAO / Our World in Data |

The full 1M-row yield file (`data/crop_yield.csv`, ~90 MB) is **not** committed; the loader falls
back to the bundled 60k stratified sample (`data/crop_yield_sample.csv`) so the pipeline runs out of
the box. Modelling uses an 80,000-row stratified-by-crop subsample so it trains on a laptop CPU.

---

## Methods

Every model is wrapped in a scikit-learn `Pipeline` (`ColumnTransformer` → estimator) so imputation,
scaling and one-hot encoding are fitted on the training folds only — the main guard against data
leakage, and what keeps the models directly comparable.

| Track | Models (simplest → most expressive) | Selection | Tuning | Metrics |
|---|---|---|---|---|
| A (classification) | Logistic Regression → Decision Tree → Random Forest → HistGradientBoosting | 5-fold CV on macro-F1 | `GridSearchCV` on the winner | Accuracy, macro P/R/F1, one-vs-rest ROC-AUC, confusion matrix |
| B (regression) | Linear Regression → Decision Tree → Random Forest → XGBoost | 5-fold CV on R² | `GridSearchCV` on the winner | R², RMSE, MAE, predicted-vs-actual, residuals |

The best model is compared against the simplest baseline with a **paired t-test** on the per-fold
scores, so "better" means statistically better, not luckier. Model behaviour is then explained with
**permutation importance** (model-agnostic, in original feature units) and **SHAP**.

> Note: XGBoost needs integer-encoded targets, so on the string-labelled classification track we use
> scikit-learn's `HistGradientBoosting` (also gradient boosting). XGBoost is used on the regression
> track, where the target is continuous.

---

## Key results

**Track A — recommendation.** The data is near-separable, so all four classifiers score highly; the
tuned **Random Forest** wins.

| Metric | Value |
|---|---|
| Test accuracy | 99.5 % |
| Macro F1 | 0.995 |
| Macro one-vs-rest ROC-AUC | 1.000 |
| Paired t-test vs Logistic Regression | p = 7.3 × 10⁻³ |

Most-important features (permutation + SHAP agree): **humidity, nitrogen, rainfall, potassium** — the
variables an agronomist would name first.

**Track B — yield.** Linear Regression already captures most of the variance; the ensembles add
little because the synthetic target is close to additive.

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Linear Regression (best) | 0.915 | 0.495 | 0.395 |

Most-important feature by far: **rainfall**, then fertiliser and irrigation use.

**Climate stress test.** Recommendations are stable to warming alone (≈0.5 % flip at +2 °C) but more
sensitive to rainfall loss (≈7.5 % flip at −30 % rain). Predicted yield falls ≈18 % under −30 %
rainfall. Rice and lentil are the most climate-fragile crops in the recommendation niche.

**Reality check.** In the dataset every crop sits at almost the same yield (cross-crop SD ≈ 0.01)
whereas real yields differ sharply (SD ≈ 1.3) — which is exactly why crop identity had near-zero
importance. The yield track is a sound *methods sandbox*; the recommendation track, built from real
Indian agro-climatic ranges, is on firmer ground.

All numbers above are regenerated into `outputs/metrics/results.json` by `run_pipeline.py`.

---

## Repository structure

```text
smart-farming-crop-ml/
├── data/
│   ├── Crop_recommendation.csv        # Track A — 2,200 rows
│   ├── crop_yield_sample.csv          # Track B — 60k stratified sample (full 1M file is git-ignored)
│   └── oecd_reference_yields.csv      # real-world yields for the reality check
├── src/                               # all reusable code (imported as the `src` package)
│   ├── config.py                      # paths, random seed, feature lists, climate scenarios
│   ├── data.py                        # load + clean both datasets, feature engineering
│   ├── models.py                      # pipelines, model zoo, hyper-parameter grids, train/test splits
│   ├── evaluate.py                    # metrics, CV comparison, paired t-test, diagnostic plots
│   ├── explain.py                     # permutation importance + SHAP
│   ├── climate.py                     # climate-scenario stress test
│   └── viz.py                         # exploratory-data-analysis figures
├── outputs/
│   ├── figures/                       # every PNG used in the report
│   ├── metrics/results.json           # all metrics, machine-readable
│   ├── results_summary.md             # headline numbers, human-readable
│   └── models/                        # trained .joblib pipelines (git-ignored, regenerated)
├── report/
│   ├── report.md                      # report template with {{tokens}}
│   ├── build_report.py                # fills tokens from results.json → report_final.md (+docx/pdf)
│   └── report_final.md                # the final report (all required sections)
├── app/
│   ├── app.py                         # Gradio app — recommender + yield tabs
│   ├── models/                        # the two .joblib copied here for Hugging Face deployment
│   ├── requirements.txt
│   └── README.md                      # Hugging Face Space card + deploy steps
├── run_pipeline.py                    # ONE command: trains both tracks, runs climate test, writes outputs
├── test_pipeline.py                   # pytest smoke tests (data contracts + fit/predict)
├── requirements.txt
├── project_proposal.md                # the Part-A proposal, kept for the record
└── README.md                          # this file
```

### Code map (what each module contains)

- **`src/config.py`** — single source of truth for folder paths, the fixed random `SEED = 42`, the
  feature lists for both tracks (`A_FEATURES`, `B_NUM_FEATURES`, `B_CAT_FEATURES`, …) and the
  `CLIMATE_SCENARIOS` list. Also `savefig()` for consistent figure output.
- **`src/data.py`** — `load_crop_recommendation()`, `load_crop_yield()` (prefers the full file, falls
  back to the sample, stratified subsampling), `clean_crop_yield()` (drops physically impossible
  negative yields, casts booleans to 0/1, engineers `rain_per_day`), plus `dtype_table()` and
  `agronomic_range_report()` for the Data section.
- **`src/models.py`** — `preprocessor_A/B()`, `pipeline_A/B()`, the `classifiers()` / `regressors()`
  model zoos, the `GRID_A` / `GRID_B` hyper-parameter grids, and `split_A/B()` (stratified vs random
  80/20 train/test).
- **`src/evaluate.py`** — `classification_metrics()`, `regression_metrics()`, `plot_confusion()`,
  `ovr_auc()`, `plot_pred_vs_actual()`, `plot_residuals()`, `cv_compare()` (per-fold CV scores) and
  `paired_ttest()`.
- **`src/explain.py`** — `permutation_fig()` (importance over raw input features) and
  `shap_summary()` (SHAP on the transformed matrix, with a graceful bar-chart fallback).
- **`src/climate.py`** — `perturb()`, `scenarios_A/B()` (flip-rate / yield-shift across scenarios),
  `fragility_A()`, and the four climate plots.
- **`src/viz.py`** — the exploratory figures used in the report (class balance, feature
  distributions, correlations, PCA, clusters, yield-by-category, etc.).
- **`run_pipeline.py`** — orchestrates everything: `run_track_A`, `run_track_B`, `run_climate`,
  `run_oecd`, then `write_summary`. Running it reproduces all figures, metrics and saved models.

---

## Reproducing the results

Requires Python 3.10+.

```bash
# 1. set up an isolated environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. train both tracks, run the climate test, write all outputs
python run_pipeline.py               # add --fast for a quicker, smaller run

# 3. (optional) run the smoke tests
pytest -q

# 4. (optional) fill the report template with the fresh numbers
python report/build_report.py        # writes report/report_final.md

# 5. (optional) launch the app locally
cd app && python app.py              # http://127.0.0.1:7860
```

`run_pipeline.py` automatically uses the full 1M-row yield file if you place it at
`data/crop_yield.csv`; otherwise it uses the bundled sample, so step 2 works with a fresh clone.

---

## Deployment

The two trained pipelines are served as a two-tab **Gradio** app on Hugging Face Spaces:
**https://huggingface.co/spaces/asternoeld/smart-farming-crop-ml**

Tab 1 takes soil + climate and returns the top-3 crops with a +2 °C climate-stability flag on the
top pick. Tab 2 takes management conditions and returns an estimated yield. The app loads the exact
`.joblib` pipelines written by `run_pipeline.py`, so the deployed behaviour matches the report. See
[`app/README.md`](app/README.md) for deployment steps.

---

## Use of AI

Following the course guideline, AI-assisted code files record the prompt used in a comment beginning
with the word `prompt`, together with notes on the manual modifications the team made. All modelling
choices, the climate-scenario design, the analysis and the conclusions are the authors' own, and
every result is reproducible with `python run_pipeline.py`.

## References

- Raschka, Liu & Mirjalili (2022). *Machine Learning with PyTorch and Scikit-Learn*. Packt.
- Ingle, A. (2021). *Crop Recommendation Dataset*. Kaggle.
- Otieno, S. *Agriculture Crop Yield*. Kaggle.
- FAO (2023). *World Food and Agriculture — Statistical Yearbook*; FAOSTAT crop yields.
- Ritchie, Roser & Rosado. *Crop Yields*. Our World in Data (FAO-based).
- Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions* (SHAP). NeurIPS.
