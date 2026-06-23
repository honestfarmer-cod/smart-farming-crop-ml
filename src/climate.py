# Use of AI: this file was drafted with an AI assistant and then reviewed
# and adapted by the team for this project.
# prompt: "write a climate sensitivity analysis that shifts temperature and scales rainfall, then measures how often the recommended crop changes and how much predicted yield changes"
# Modifications: <describe the changes you made, e.g. parameter values, fixes>

"""
Climate sensitivity analysis.

Takes the trained models and re-runs them on inputs where temperature is shifted
and rainfall is scaled. For the recommender it measures how often the suggested
crop changes; for the yield model it measures how much predicted yield changes.
Crops are then ranked from least to most sensitive.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C


# ---------------------------------------------------------------------------
# perturbation
# ---------------------------------------------------------------------------
def perturb(X: pd.DataFrame, d_temp: float, rain_mult: float,
            temp_col: str, rain_col: str, days_col: str | None = None) -> pd.DataFrame:
    """Return a copy of X with temperature shifted and rainfall scaled."""
    Xp = X.copy()
    Xp[temp_col] = Xp[temp_col] + d_temp
    Xp[rain_col] = Xp[rain_col] * rain_mult
    if days_col and "rain_per_day" in Xp.columns:        # keep engineered feature consistent
        Xp["rain_per_day"] = Xp[rain_col] / Xp[days_col].clip(lower=1)
    return Xp


# ---------------------------------------------------------------------------
# Track A — recommendation flips
# ---------------------------------------------------------------------------
def scenarios_A(pipe, X_test) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pipe.predict(X_test)
    overall, per_crop = [], []
    for label, dT, rm in C.CLIMATE_SCENARIOS:
        pred = pipe.predict(perturb(X_test, dT, rm, "temperature", "rainfall"))
        flipped = pred != base
        overall.append({"scenario": label, "d_temp": dT, "rain_mult": rm,
                        "flip_rate": float(flipped.mean())})
        s = pd.Series(flipped).groupby(pd.Series(base)).mean()
        s.name = label
        per_crop.append(s)
    return pd.DataFrame(overall), pd.concat(per_crop, axis=1)


def fragility_A(per_crop: pd.DataFrame, scenario="+2°C") -> pd.Series:
    """Per-crop flip rate under one warming scenario, fragile first."""
    return per_crop[scenario].sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Track B — yield shifts
# ---------------------------------------------------------------------------
def scenarios_B(pipe, X_test) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pipe.predict(X_test)
    crops = X_test["Crop"].to_numpy()
    overall, per_crop = [], []
    for label, dT, rm in C.CLIMATE_SCENARIOS:
        pred = pipe.predict(perturb(X_test, dT, rm, "Temperature_Celsius",
                                    "Rainfall_mm", days_col="Days_to_Harvest"))
        delta = pred - base
        overall.append({"scenario": label, "d_temp": dT, "rain_mult": rm,
                        "mean_yield_change": float(delta.mean()),
                        "pct_change": float(delta.mean() / base.mean() * 100)})
        s = pd.Series(delta).groupby(pd.Series(crops)).mean()
        s.name = label
        per_crop.append(s)
    return pd.DataFrame(overall), pd.concat(per_crop, axis=1)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def plot_warming_curve_A(overall_A, fname="C_flip_rate_vs_warming.png"):
    warm = overall_A[overall_A["rain_mult"] == 1.0].sort_values("d_temp")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(warm["d_temp"], warm["flip_rate"] * 100, "o-", color="#c44e52")
    ax.set(title="Track A — recommendation instability under warming",
           xlabel="temperature change (°C)", ylabel="recommendations that flip (%)")
    return C.savefig(fig, fname)


def plot_fragility_A(frag, fname="C_crop_fragility.png"):
    fig, ax = plt.subplots(figsize=(9, 6))
    (frag * 100).sort_values().plot.barh(ax=ax, color="#dd8452")
    ax.set(title="Track A — crop fragility (flip rate at +2°C)",
           xlabel="recommendations that flip (%)")
    return C.savefig(fig, fname)


def plot_warming_curve_B(overall_B, fname="C_yield_vs_warming.png"):
    warm = overall_B[overall_B["rain_mult"] == 1.0].sort_values("d_temp")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(warm["d_temp"], warm["pct_change"], "o-", color="#4c72b0")
    ax.axhline(0, color="k", lw=1)
    ax.set(title="Track B — predicted yield change under warming",
           xlabel="temperature change (°C)", ylabel="mean yield change (%)")
    return C.savefig(fig, fname)


def plot_yield_fragility_B(per_crop_B, scenario="+2°C, -20% rain",
                           fname="C_yield_fragility.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    per_crop_B[scenario].sort_values().plot.bar(ax=ax, color="#55a868")
    ax.axhline(0, color="k", lw=1)
    ax.set(title=f"Track B — yield change per crop ({scenario})",
           ylabel="yield change (t/ha)", xlabel="")
    return C.savefig(fig, fname)
