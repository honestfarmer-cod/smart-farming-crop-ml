"""Smoke tests for the pipeline. Run with:  pytest -q

These check the data contracts and that the pipelines fit/predict on a small
slice. They are fast and meant as a sanity net, not exhaustive coverage.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from src import config as C, data, models, evaluate as ev, climate


# --- data --------------------------------------------------------------------
def test_track_a_loads_balanced():
    df = data.load_crop_recommendation()
    assert df.shape == (2200, 8)
    assert df["label"].nunique() == 22
    assert df["label"].value_counts().nunique() == 1          # perfectly balanced


def test_cleaning_removes_negative_yields():
    raw = data.load_crop_yield(prefer_full=False, n_rows=5000)
    clean, rep = data.clean_crop_yield(raw)
    assert (clean[C.B_TARGET] >= 0).all()
    assert rep["rows_after"] == len(clean)
    assert "rain_per_day" in clean.columns


def test_dtype_table():
    df = data.load_crop_recommendation()
    t = data.dtype_table(df)
    assert set(["column", "dtype", "type", "n_unique", "n_missing"]).issubset(t.columns)


# --- models ------------------------------------------------------------------
def test_pipeline_a_fits_and_predicts():
    df = data.load_crop_recommendation().groupby("label").head(20)   # 22*20 rows
    X_tr, X_te, y_tr, y_te = models.split_A(df)
    pipe = models.pipeline_A(models.classifiers()["tree"]).fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    assert len(pred) == len(y_te)
    m = ev.classification_metrics(y_te, pred)
    assert 0.0 <= m["accuracy"] <= 1.0


def test_pipeline_b_fits_and_predicts():
    raw = data.load_crop_yield(prefer_full=False, n_rows=4000)
    clean, _ = data.clean_crop_yield(raw)
    X_tr, X_te, y_tr, y_te = models.split_B(clean)
    pipe = models.pipeline_B(models.regressors()["linreg"]).fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    assert len(pred) == len(y_te)
    assert np.isfinite(pred).all()


# --- climate -----------------------------------------------------------------
def test_perturb_shifts_inputs():
    df = data.load_crop_recommendation().head(50)
    X = df[C.A_FEATURES]
    Xp = climate.perturb(X, 2.0, 0.8, "temperature", "rainfall")
    assert np.allclose(Xp["temperature"], X["temperature"] + 2.0)
    assert np.allclose(Xp["rainfall"], X["rainfall"] * 0.8)
