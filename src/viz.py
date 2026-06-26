# Use of AI: this file was drafted with an AI assistant and then reviewed
# and adapted by the team for this project.
# prompt: "create exploratory data analysis plots for both datasets using matplotlib and seaborn"
# Modifications: 

"""
Exploratory data analysis plots.

Uses only pandas, numpy, matplotlib and seaborn, so the figures can be regenerated
without the modelling libraries. The PCA and k-means used here are small NumPy
versions intended only for the exploratory plots.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from . import config as C

sns.set_theme(style="whitegrid", context="notebook")


# ---------------------------------------------------------------------------
# tiny NumPy unsupervised helpers (for exploratory ordination only)
# ---------------------------------------------------------------------------
def _pca_2d(X: np.ndarray):
    Xc = (X - X.mean(0)) / X.std(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = Xc @ Vt[:2].T
    var = (S ** 2) / (S ** 2).sum()
    return scores, var[:2], Vt[:2]


def _kmeans(X: np.ndarray, k: int, iters: int = 100, seed: int = C.SEED):
    rng = np.random.default_rng(seed)
    Xc = (X - X.mean(0)) / X.std(0)
    centers = Xc[rng.choice(len(Xc), k, replace=False)]
    for _ in range(iters):
        d = ((Xc[:, None, :] - centers[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        new = np.array([Xc[lab == j].mean(0) if (lab == j).any() else centers[j]
                        for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    return lab


# ---------------------------------------------------------------------------
# Track A figures
# ---------------------------------------------------------------------------
def a_class_balance(df, fname="A_class_balance.png"):
    fig, ax = plt.subplots(figsize=(9, 5))
    df[C.A_TARGET].value_counts().sort_index().plot.bar(ax=ax, color="#4c8c4a")
    ax.set(title="Track A — samples per crop (perfectly balanced, 100 each)",
           xlabel="crop", ylabel="count")
    plt.xticks(rotation=60, ha="right")
    return C.savefig(fig, fname)


def a_feature_distributions(df, fname="A_feature_distributions.png"):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), C.A_FEATURES):
        sns.histplot(df[col], kde=True, ax=ax, color="#4c72b0")
        ax.set_title(C.A_PRETTY[col]); ax.set_xlabel("")
    axes.ravel()[-1].axis("off")
    fig.suptitle("Track A — feature distributions", y=1.02, fontsize=13)
    return C.savefig(fig, fname)


def a_correlation(df, fname="A_correlation.png"):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df[C.A_FEATURES].corr(), annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax)
    ax.set_title("Track A — feature correlation")
    return C.savefig(fig, fname)


def a_boxplots_by_crop(df, fname="A_boxplots_by_crop.png"):
    fig, axes = plt.subplots(4, 2, figsize=(15, 16))
    order = sorted(df[C.A_TARGET].unique())
    for ax, col in zip(axes.ravel(), C.A_FEATURES):
        sns.boxplot(data=df, x=C.A_TARGET, y=col, order=order, ax=ax,
                    color="#8da0cb", fliersize=1)
        ax.set_title(C.A_PRETTY[col]); ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=90)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Track A — how each crop's requirements differ", y=1.0, fontsize=13)
    return C.savefig(fig, fname)


def a_pca(df, fname="A_pca.png"):
    X = df[C.A_FEATURES].to_numpy(float)
    scores, var, _ = _pca_2d(X)
    fig, ax = plt.subplots(figsize=(9, 7))
    crops = df[C.A_TARGET].to_numpy()
    for crop in sorted(np.unique(crops)):
        m = crops == crop
        ax.scatter(scores[m, 0], scores[m, 1], s=12, label=crop, alpha=.7)
    ax.set(title=f"Track A — PCA ordination (PC1 {var[0]*100:.0f}%, PC2 {var[1]*100:.0f}%)",
           xlabel="PC1", ylabel="PC2")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, ncol=1)
    return C.savefig(fig, fname)


def a_clusters(df, k=5, fname="A_clusters.png"):
    X = df[C.A_FEATURES].to_numpy(float)
    scores, var, _ = _pca_2d(X)
    lab = _kmeans(X, k)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(scores[:, 0], scores[:, 1], c=lab, cmap="tab10", s=14, alpha=.8)
    ax.set(title=f"Track A — k-means ({k} clusters) in PCA space",
           xlabel="PC1", ylabel="PC2")
    fig.colorbar(sc, ax=ax, label="cluster")
    return C.savefig(fig, fname)


# ---------------------------------------------------------------------------
# Track B figures
# ---------------------------------------------------------------------------
def b_yield_before_after(raw, clean, fname="B_yield_before_after.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    sns.histplot(raw[C.B_TARGET], bins=60, ax=axes[0], color="#c44e52")
    axes[0].axvline(0, color="k", ls="--", lw=1)
    axes[0].set_title(f"Before cleaning — {int((raw[C.B_TARGET] < 0).sum())} "
                      f"impossible negative yields")
    axes[0].set_xlabel("Yield (t/ha)")
    sns.histplot(clean[C.B_TARGET], bins=60, ax=axes[1], color="#55a868")
    axes[1].set_title("After cleaning — negatives removed")
    axes[1].set_xlabel("Yield (t/ha)")
    fig.suptitle("Track B — target distribution before vs after cleaning", y=1.03)
    return C.savefig(fig, fname)


def b_yield_by_crop(df, fname="B_yield_by_crop.png"):
    fig, ax = plt.subplots(figsize=(9, 5))
    order = df.groupby("Crop")[C.B_TARGET].median().sort_values().index
    sns.boxplot(data=df, x="Crop", y=C.B_TARGET, order=order, ax=ax, color="#64b5cd")
    ax.set(title="Track B — yield by crop", xlabel="", ylabel="Yield (t/ha)")
    return C.savefig(fig, fname)


def b_yield_by_category(df, fname="B_yield_by_category.png"):
    cats = ["Region", "Soil_Type", "Weather_Condition"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, col in zip(axes, cats):
        sns.boxplot(data=df, x=col, y=C.B_TARGET, ax=ax, color="#b5b5e0")
        ax.set(title=f"Yield by {col}", xlabel="", ylabel="Yield (t/ha)")
        ax.tick_params(axis="x", rotation=30)
    return C.savefig(fig, fname)


def b_yield_vs_numeric(df, fname="B_yield_vs_numeric.png"):
    cols = ["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, col in zip(axes, cols):
        ax.hexbin(df[col], df[C.B_TARGET], gridsize=40, cmap="viridis", mincnt=1)
        ax.set(title=f"Yield vs {col}", xlabel=col, ylabel="Yield (t/ha)")
    return C.savefig(fig, fname)


def b_management_effect(df, fname="B_management_effect.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col in zip(axes, ["Fertilizer_Used", "Irrigation_Used"]):
        sns.boxplot(data=df, x=col, y=C.B_TARGET, ax=ax, color="#ccb974")
        ax.set(title=f"Effect of {col.replace('_', ' ')}", xlabel="", ylabel="Yield (t/ha)")
    return C.savefig(fig, fname)


def b_correlation(df, fname="B_correlation.png"):
    num = ["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest",
           "Fertilizer_Used", "Irrigation_Used", C.B_TARGET]
    d = df[num].copy()
    for b in ["Fertilizer_Used", "Irrigation_Used"]:
        d[b] = d[b].astype(int)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(d.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Track B — numeric correlation with yield")
    return C.savefig(fig, fname)
