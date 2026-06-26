# Use of AI: this file was drafted with an AI assistant and then reviewed
# and adapted by the team for this project.
# prompt: "compute permutation importance and a SHAP summary for the fitted pipelines"
# Modifications: 

"""
Model explainability.

Computes permutation importance on the fitted pipeline using the original input
features, and a SHAP summary on the transformed features. If SHAP is not installed
or its plot fails, the function falls back to a mean-importance bar chart.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance

from . import config as C


def permutation_fig(pipe, X_test, y_test, feature_names, title, fname,
                    scoring=None, n_repeats=10):
    """Permutation importance over the raw input features."""
    r = permutation_importance(pipe, X_test, y_test, scoring=scoring,
                               n_repeats=n_repeats, random_state=C.SEED, n_jobs=-1)
    imp = pd.Series(r.importances_mean, index=feature_names).sort_values()
    err = pd.Series(r.importances_std, index=feature_names).loc[imp.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.plot.barh(ax=ax, xerr=err, color="#4c72b0")
    ax.set(title=title, xlabel="drop in score when feature is shuffled")
    path = C.savefig(fig, fname)
    return imp[::-1], path


def _transform(pipe, X):
    """Return (transformed matrix, output feature names, final estimator)."""
    prep = pipe[:-1]
    est = pipe[-1]
    Xt = prep.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    try:
        names = list(prep.get_feature_names_out())
    except Exception:
        names = [f"f{i}" for i in range(Xt.shape[1])]
    return np.asarray(Xt), names, est


def shap_summary(pipe, X_sample, title, fname, max_display=15):
    """SHAP global importance; falls back to a bar chart, then to None."""
    try:
        import shap
    except Exception as e:                                    # pragma: no cover
        print("SHAP not installed:", e)
        return None
    Xt, names, est = _transform(pipe, X_sample)
    try:
        explainer = shap.Explainer(est, Xt, feature_names=names)
        sv = explainer(Xt, check_additivity=False)
        plt.figure()
        shap.summary_plot(sv, features=Xt, feature_names=names, show=False,
                          max_display=max_display)
        fig = plt.gcf()
        fig.suptitle(title, y=1.02)
        path = C.savefig(fig, fname)
        plt.close(fig)
        return path
    except Exception as e:                                    # pragma: no cover
        print("SHAP summary_plot fell back to a bar chart:", e)
        try:
            vals = np.abs(sv.values)
        except Exception:
            return None
        while vals.ndim > 2:
            vals = vals.mean(axis=-1)
        imp = pd.Series(vals.mean(axis=0), index=names).sort_values().tail(max_display)
        fig, ax = plt.subplots(figsize=(8, 5))
        imp.plot.barh(ax=ax, color="#8172b3")
        ax.set(title=title + " (mean |SHAP|)", xlabel="mean |SHAP value|")
        path = C.savefig(fig, fname)
        plt.close(fig)
        return path
