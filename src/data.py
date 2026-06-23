# Use of AI: this file was drafted with an AI assistant and then reviewed
# and adapted by the team for this project.
# prompt: "write functions to load and clean the crop-recommendation and crop-yield datasets, remove impossible negative yields, and add a rainfall-per-day feature"
# Modifications: <describe the changes you made, e.g. parameter values, fixes>

"""
Loading and cleaning functions for the two datasets.

This module uses only pandas and numpy. Encoding, scaling and the train/test split
are done separately in models.py inside a scikit-learn Pipeline, so those steps are
fitted on the training data only.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------------------
# Track A — crop recommendation
# ---------------------------------------------------------------------------
def load_crop_recommendation() -> pd.DataFrame:
    """Read the crop-recommendation table (2,200 rows, 22 balanced classes)."""
    df = pd.read_csv(C.CROP_CSV)
    return df


def agronomic_range_report(df: pd.DataFrame) -> pd.DataFrame:
    """Flag values that fall outside agronomically sensible ranges.

    These are sanity bounds, not hard physical limits; they tell us whether the
    (augmented) data behaves like real soil/climate measurements.
    """
    bounds = {
        "N": (0, 200), "P": (0, 200), "K": (0, 250),
        "temperature": (-5, 50), "humidity": (0, 100),
        "ph": (3.0, 10.0), "rainfall": (10, 350),
    }
    rows = []
    for col, (lo, hi) in bounds.items():
        out = int(((df[col] < lo) | (df[col] > hi)).sum())
        rows.append({"feature": col, "min": df[col].min(), "max": df[col].max(),
                     "plausible_range": f"{lo}–{hi}", "n_out_of_range": out})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Track B — crop yield
# ---------------------------------------------------------------------------
def load_crop_yield(prefer_full: bool = True, n_rows: int | None = None,
                    seed: int = C.SEED) -> pd.DataFrame:
    """Load the yield data.

    Uses the full 1M-row file when it is present, otherwise the 60k stratified
    sample that ships in the repo. ``n_rows`` optionally draws a further
    stratified-by-crop subsample so the heavier models train quickly on a CPU.
    """
    path = C.YIELD_FULL_CSV if (prefer_full and C.YIELD_FULL_CSV.exists()) else C.YIELD_SAMPLE_CSV
    df = pd.read_csv(path)
    if n_rows is not None and n_rows < len(df):
        per = max(1, n_rows // df["Crop"].nunique())
        parts = [d.sample(min(len(d), per), random_state=seed)
                 for _, d in df.groupby("Crop")]
        df = pd.concat(parts).reset_index(drop=True)
    return df


def clean_crop_yield(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the yield table and return a short before/after report.

    The dataset contains some negative yields, which are not physically possible
    and come from the way it was generated; those rows are removed. Boolean columns
    are converted to integers so they pass through the numeric part of the pipeline.
    """
    n0 = len(df)
    neg = int((df[C.B_TARGET] < 0).sum())
    out = df[df[C.B_TARGET] >= 0].copy()
    for b in C.B_BOOL_FEATURES:
        out[b] = out[b].astype(int)
    # engineered feature: rainfall received per growing day
    out["rain_per_day"] = out["Rainfall_mm"] / out["Days_to_Harvest"].clip(lower=1)
    report = {
        "rows_before": n0,
        "negative_yields_removed": neg,
        "rows_after": len(out),
        "engineered_features": ["rain_per_day"],
    }
    return out, report


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def dtype_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compact data-type / cardinality summary (used in the report's Data section)."""
    rows = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            kind = "boolean"
        elif pd.api.types.is_numeric_dtype(s):
            kind = "numeric (continuous)" if s.nunique() > 20 else "numeric (discrete)"
        else:
            kind = "categorical"
        rows.append({"column": col, "dtype": str(s.dtype), "type": kind,
                     "n_unique": int(s.nunique()), "n_missing": int(s.isna().sum())})
    return pd.DataFrame(rows)
