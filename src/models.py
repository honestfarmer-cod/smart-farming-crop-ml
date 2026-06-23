# Use of AI: this file was drafted with an AI assistant and then reviewed
# and adapted by the team for this project.
# prompt: "build scikit-learn pipelines and a set of models (logistic regression, decision tree, random forest, XGBoost) for the classification and regression tracks, with preprocessing inside each pipeline and small hyper-parameter grids"
# Modifications: <describe the changes you made, e.g. parameter values, fixes>

"""
Model definitions and scikit-learn pipelines for both tracks.

Each model is wrapped in a Pipeline together with its preprocessing, so imputation,
scaling and one-hot encoding are fitted during cross-validation on the training
folds only. XGBoost is used when installed; otherwise the code uses scikit-learn's
HistGradientBoosting so the project still runs.
"""
from __future__ import annotations
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor)
from sklearn.model_selection import train_test_split

from . import config as C

# Optional XGBoost ------------------------------------------------------------
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:                                    # pragma: no cover
    HAS_XGB = False


# ---------------------------------------------------------------------------
# Track A — preprocessing + classifiers
# ---------------------------------------------------------------------------
def preprocessor_A() -> ColumnTransformer:
    """All seven Track A inputs are numeric, so impute missing values then standardise."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    return ColumnTransformer([("num", numeric, C.A_FEATURES)], remainder="drop")


def classifiers() -> dict:
    """The course classifiers, from simplest to most expressive."""
    zoo = {
        "logreg": LogisticRegression(max_iter=2000, random_state=C.SEED),
        "tree":   DecisionTreeClassifier(random_state=C.SEED),
        "rf":     RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                         random_state=C.SEED),
    }
    # XGBoost's classifier needs integer-encoded targets, but our crop labels are
    # strings, so for the classification track we use scikit-learn's
    # HistGradientBoosting (also a gradient-boosting method). XGBoost is used in the
    # regression track below, where the target is continuous.
    zoo["hgb"] = HistGradientBoostingClassifier(random_state=C.SEED)
    return zoo


def pipeline_A(estimator) -> Pipeline:
    return Pipeline([("prep", preprocessor_A()), ("clf", estimator)])


# hyper-parameter grids for GridSearchCV
GRID_A = {
    "tree": {"clf__max_depth": [4, 6, 8, 12, None],
             "clf__min_samples_leaf": [1, 2, 5]},
    "rf":   {"clf__n_estimators": [200, 400],
             "clf__max_depth": [None, 10, 20]},
    "xgb":  {"clf__max_depth": [3, 4, 6],
             "clf__learning_rate": [0.05, 0.1, 0.2]},
    "hgb":  {"clf__max_depth": [None, 4, 8],
             "clf__learning_rate": [0.05, 0.1]},
    "logreg": {"clf__C": [0.1, 1.0, 10.0]},
}


# ---------------------------------------------------------------------------
# Track B — preprocessing + regressors
# ---------------------------------------------------------------------------
def preprocessor_B() -> ColumnTransformer:
    """Scale the numeric columns and one-hot encode the four categorical columns.

    Boolean columns were converted to 0/1 during cleaning, so they go through the
    numeric branch. rain_per_day is the engineered feature added in data.py.
    """
    numeric = C.B_NUM_FEATURES + C.B_BOOL_FEATURES + ["rain_per_day"]
    num_t = Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())])
    cat_t = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_t, numeric),
                              ("cat", cat_t, C.B_CAT_FEATURES)], remainder="drop")


def regressors() -> dict:
    zoo = {
        "linreg": LinearRegression(),
        "tree":   DecisionTreeRegressor(max_depth=12, random_state=C.SEED),
        "rf":     RandomForestRegressor(n_estimators=200, n_jobs=-1,
                                        random_state=C.SEED),
    }
    if HAS_XGB:
        zoo["xgb"] = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.08,
                                  subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
                                  random_state=C.SEED)
    else:
        zoo["hgb"] = HistGradientBoostingRegressor(random_state=C.SEED)
    return zoo


def pipeline_B(estimator) -> Pipeline:
    return Pipeline([("prep", preprocessor_B()), ("reg", estimator)])


GRID_B = {
    "tree": {"reg__max_depth": [6, 10, 14, None]},
    "rf":   {"reg__n_estimators": [200, 400],
             "reg__max_depth": [None, 16]},
    "xgb":  {"reg__max_depth": [4, 6, 8],
             "reg__learning_rate": [0.05, 0.1]},
    "hgb":  {"reg__max_depth": [None, 6, 10]},
}


# ---------------------------------------------------------------------------
# splitting helpers (Data Organization section of the report)
# ---------------------------------------------------------------------------
def split_A(df):
    """Stratified 80/20 train/test. k-fold CV on the train part is the validation."""
    X = df[C.A_FEATURES]
    y = df[C.A_TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=C.SEED)


def split_B(df):
    """Random 80/20 train/test for yield regression (CV on train = validation)."""
    feats = C.B_NUM_FEATURES + C.B_BOOL_FEATURES + ["rain_per_day"] + C.B_CAT_FEATURES
    X = df[feats]
    y = df[C.B_TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=C.SEED)
