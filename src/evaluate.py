# Use of AI: this file was drafted with an AI assistant and then reviewed
# and adapted by the team for this project.
# prompt: "write evaluation helpers: accuracy and macro F1, confusion matrix, one-vs-rest ROC-AUC, R2/RMSE/MAE, predicted-vs-actual and residual plots, a cross-validation comparison and a paired t-test"
# Modifications: <describe the changes you made, e.g. parameter values, fixes>

"""
Evaluation metrics and diagnostic plots.

Classification: accuracy, macro precision/recall/F1, confusion matrix and
one-vs-rest ROC-AUC. Regression: R-squared, RMSE, MAE, predicted-vs-actual and
residual plots. Models are compared with k-fold cross-validation and a paired
t-test on the per-fold scores.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score, classification_report,
                             r2_score, mean_squared_error, mean_absolute_error)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

from . import config as C


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
def classification_metrics(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                                 zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": p, "recall_macro": r, "f1_macro": f}


def plot_confusion(y_true, y_pred, labels, title, fname):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(11, 10))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=90, values_format="d")
    ax.set_title(title)
    return C.savefig(fig, fname)


def ovr_auc(pipe, X_test, y_test, fname="A_roc_auc_per_class.png"):
    """One-vs-rest ROC-AUC: macro score + a per-class AUC bar chart."""
    classes = list(pipe.classes_)                      # column order of predict_proba
    proba = pipe.predict_proba(X_test)
    y_bin = label_binarize(y_test, classes=classes)
    # multi_class OvR expects 1-D true labels + (n, n_classes) scores
    macro = roc_auc_score(y_test, proba, average="macro", multi_class="ovr", labels=classes)
    per_class = {c: roc_auc_score(y_bin[:, i], proba[:, i]) for i, c in enumerate(classes)}
    fig, ax = plt.subplots(figsize=(9, 6))
    s = pd.Series(per_class).sort_values()
    s.plot.barh(ax=ax, color="#4c8c4a")
    ax.set(title=f"Track A — one-vs-rest ROC-AUC per crop (macro = {macro:.3f})",
           xlabel="ROC-AUC", xlim=(min(0.9, s.min() - 0.01), 1.001))
    path = C.savefig(fig, fname)
    return macro, per_class, path


def text_report(y_true, y_pred):
    return classification_report(y_true, y_pred, zero_division=0)


# ---------------------------------------------------------------------------
# regression
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"r2": r2_score(y_true, y_pred), "rmse": rmse,
            "mae": mean_absolute_error(y_true, y_pred)}


def plot_pred_vs_actual(y_true, y_pred, title, fname):
    """Scatter of predicted vs actual values with a y = x reference line."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_true, y_pred, s=6, alpha=0.25, color="#4c72b0")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="perfect prediction")
    r2 = r2_score(y_true, y_pred)
    ax.set(title=f"{title}\nR² = {r2:.3f}", xlabel="Actual yield (t/ha)",
           ylabel="Predicted yield (t/ha)")
    ax.legend()
    return C.savefig(fig, fname)


def plot_residuals(y_true, y_pred, title, fname):
    resid = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].scatter(y_pred, resid, s=6, alpha=0.25, color="#c44e52")
    axes[0].axhline(0, color="k", lw=1)
    axes[0].set(title="Residuals vs predicted", xlabel="Predicted", ylabel="Residual")
    axes[1].hist(resid, bins=50, color="#55a868")
    axes[1].set(title="Residual distribution", xlabel="Residual", ylabel="count")
    fig.suptitle(title, y=1.02)
    return C.savefig(fig, fname)


# ---------------------------------------------------------------------------
# model comparison with cross-validation + paired test
# ---------------------------------------------------------------------------
def cv_compare(make_pipeline, zoo, X, y, scoring, task="clf", n_splits=5):
    """Return a tidy DataFrame of per-fold CV scores for every model in ``zoo``."""
    cv = (StratifiedKFold(n_splits, shuffle=True, random_state=C.SEED) if task == "clf"
          else KFold(n_splits, shuffle=True, random_state=C.SEED))
    rows = []
    for name, est in zoo.items():
        scores = cross_val_score(make_pipeline(est), X, y, cv=cv, scoring=scoring,
                                 n_jobs=-1)
        for k, s in enumerate(scores):
            rows.append({"model": name, "fold": k, "score": s})
    return pd.DataFrame(rows)


def plot_cv_box(cv_df, scoring, title, fname):
    fig, ax = plt.subplots(figsize=(8, 5))
    order = cv_df.groupby("model")["score"].mean().sort_values().index
    data = [cv_df.loc[cv_df.model == m, "score"].values for m in order]
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(list(order))
    ax.set(title=title, ylabel=scoring, xlabel="model")
    return C.savefig(fig, fname)


def paired_ttest(cv_df, a, b):
    """Paired t-test on per-fold scores of two models (model comparison)."""
    sa = cv_df.loc[cv_df.model == a].sort_values("fold")["score"].values
    sb = cv_df.loc[cv_df.model == b].sort_values("fold")["score"].values
    t, p = stats.ttest_rel(sa, sb)
    return {"model_a": a, "model_b": b, "mean_a": sa.mean(), "mean_b": sb.mean(),
            "t_stat": float(t), "p_value": float(p)}
