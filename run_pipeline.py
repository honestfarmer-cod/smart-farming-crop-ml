"""
Main entry point for the Smart Farming project.

Running this script reproduces the whole analysis in one command: it trains the
crop-recommendation classifier (Track A) and the crop-yield regressor (Track B),
runs the climate sensitivity analysis, and writes all figures, metrics and trained
models into the outputs/ folder.

Usage:
    python run_pipeline.py          Train on the full data, or the bundled sample
                                    if the full yield file is not present.
    python run_pipeline.py --fast   Use a smaller yield subsample for a quick run.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold

from src import config as C, data, viz, models, evaluate as ev, explain, climate


def _jsonable(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, dict): return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_jsonable(v) for v in o]
    if isinstance(o, pd.Series): return _jsonable(o.to_dict())
    return o


# ---------------------------------------------------------------------------
def run_track_A(results):
    print("\n=== TRACK A — crop recommendation (classification) ===")
    df = data.load_crop_recommendation()
    # exploratory data analysis figures
    viz.a_class_balance(df); viz.a_feature_distributions(df); viz.a_correlation(df)
    viz.a_boxplots_by_crop(df); viz.a_pca(df); viz.a_clusters(df)

    X_tr, X_te, y_tr, y_te = models.split_A(df)
    labels = sorted(df[C.A_TARGET].unique())

    # cross-validated model comparison on the training set
    cv = ev.cv_compare(models.pipeline_A, models.classifiers(), X_tr, y_tr,
                       scoring="f1_macro", task="clf")
    ev.plot_cv_box(cv, "f1_macro", "Track A — 5-fold CV (macro-F1)", "A_cv_compare.png")
    cv_mean = cv.groupby("model")["score"].mean().sort_values()
    best_name = cv_mean.index[-1]
    print("CV macro-F1:\n", cv_mean.round(4).to_string())
    print("best by CV:", best_name)

    # tune the best model
    base = models.classifiers()[best_name]
    grid = models.GRID_A.get(best_name, {})
    search = GridSearchCV(models.pipeline_A(base), grid,
                          cv=StratifiedKFold(5, shuffle=True, random_state=C.SEED),
                          scoring="f1_macro", n_jobs=-1)
    search.fit(X_tr, y_tr)
    best = search.best_estimator_
    print("best params:", search.best_params_)

    # evaluate on the held-out test set
    y_pred = best.predict(X_te)
    m = ev.classification_metrics(y_te, y_pred)
    ev.plot_confusion(y_te, y_pred, labels,
                      f"Track A — confusion matrix ({best_name}, tuned)",
                      "A_confusion.png")
    macro_auc, per_auc, _ = ev.ovr_auc(best, X_te, y_te)
    imp, _ = explain.permutation_fig(best, X_te, y_te, C.A_FEATURES,
                                     "Track A — permutation importance",
                                     "A_permutation.png", scoring="accuracy")
    explain.shap_summary(best, X_te.sample(min(400, len(X_te)), random_state=C.SEED),
                         "Track A — SHAP summary", "A_shap.png")

    # paired test: best vs the simplest baseline
    pt = ev.paired_ttest(cv, "logreg", best_name)

    joblib.dump({"pipeline": best, "classes": labels, "features": C.A_FEATURES},
                C.MODELS / "track_A_classifier.joblib")
    results["track_A"] = {"best_model": best_name, "best_params": search.best_params_,
                          "cv_macro_f1": _jsonable(cv_mean), "test_metrics": _jsonable(m),
                          "macro_roc_auc": macro_auc,
                          "permutation_importance": _jsonable(imp),
                          "paired_ttest_logreg_vs_best": pt}
    return best, X_te, y_te


# ---------------------------------------------------------------------------
def run_track_B(results, fast=False):
    print("\n=== TRACK B — yield prediction (regression) ===")
    n = 30_000 if fast else C.B_MODEL_ROWS
    raw = data.load_crop_yield(prefer_full=True, n_rows=n)
    clean, rep = data.clean_crop_yield(raw)
    print("cleaning:", rep)
    viz.b_yield_before_after(raw, clean); viz.b_yield_by_crop(clean)
    viz.b_yield_by_category(clean); viz.b_yield_vs_numeric(clean)
    viz.b_management_effect(clean); viz.b_correlation(clean)

    X_tr, X_te, y_tr, y_te = models.split_B(clean)

    cv = ev.cv_compare(models.pipeline_B, models.regressors(), X_tr, y_tr,
                       scoring="r2", task="reg")
    ev.plot_cv_box(cv, "r2", "Track B — 5-fold CV (R²)", "B_cv_compare.png")
    cv_mean = cv.groupby("model")["score"].mean().sort_values()
    best_name = cv_mean.index[-1]
    print("CV R²:\n", cv_mean.round(4).to_string())

    base = models.regressors()[best_name]
    grid = models.GRID_B.get(best_name, {})
    search = GridSearchCV(models.pipeline_B(base), grid,
                          cv=KFold(5, shuffle=True, random_state=C.SEED),
                          scoring="r2", n_jobs=-1)
    search.fit(X_tr, y_tr)
    best = search.best_estimator_

    # baseline (linear) vs best, both on the test set, with pred-vs-actual plots
    lin = models.pipeline_B(models.regressors()["linreg"]).fit(X_tr, y_tr)
    for tag, model in [("linreg", lin), (best_name, best)]:
        yp = model.predict(X_te)
        mm = ev.regression_metrics(y_te, yp)
        ev.plot_pred_vs_actual(y_te.values, yp,
                               f"Track B — {tag}: predicted vs actual",
                               f"B_pred_vs_actual_{tag}.png")
        ev.plot_residuals(y_te.values, yp, f"Track B — {tag}: residuals",
                          f"B_residuals_{tag}.png")
        # stable alias for the report (independent of which model wins)
        if tag == best_name:
            ev.plot_pred_vs_actual(y_te.values, yp,
                                   f"Track B — {tag}: predicted vs actual",
                                   "B_pred_vs_actual_best.png")
        results.setdefault("track_B_models", {})[tag] = _jsonable(mm)
        print(f"{tag}: ", {k: round(v, 4) for k, v in mm.items()})

    imp, _ = explain.permutation_fig(
        best, X_te, y_te,
        C.B_NUM_FEATURES + C.B_BOOL_FEATURES + ["rain_per_day"] + C.B_CAT_FEATURES,
        "Track B — permutation importance", "B_permutation.png", scoring="r2")
    explain.shap_summary(best, X_te.sample(min(500, len(X_te)), random_state=C.SEED),
                         "Track B — SHAP summary", "B_shap.png")

    joblib.dump({"pipeline": best, "target": C.B_TARGET}, C.MODELS / "track_B_regressor.joblib")
    results["track_B"] = {"best_model": best_name, "best_params": search.best_params_,
                          "cleaning": rep, "cv_r2": _jsonable(cv_mean),
                          "permutation_importance": _jsonable(imp)}
    return best, X_te, y_te


# ---------------------------------------------------------------------------
def run_climate(results, clf, Xa, reg, Xb):
    print("\n=== CLIMATE-SCENARIO STRESS TEST ===")
    ov_A, pc_A = climate.scenarios_A(clf, Xa)
    climate.plot_warming_curve_A(ov_A)
    frag = climate.fragility_A(pc_A, "+2°C")
    climate.plot_fragility_A(frag)
    print("Track A flip rates:\n", ov_A.round(3).to_string(index=False))

    ov_B, pc_B = climate.scenarios_B(reg, Xb)
    climate.plot_warming_curve_B(ov_B)
    climate.plot_yield_fragility_B(pc_B)
    print("Track B yield shifts:\n", ov_B.round(3).to_string(index=False))

    results["climate"] = {
        "trackA_flip_rates": _jsonable(ov_A.to_dict("records")),
        "trackA_fragility_plus2C": _jsonable(frag),
        "trackB_yield_shifts": _jsonable(ov_B.to_dict("records"))}


# ---------------------------------------------------------------------------
def write_summary(results):
    (C.METRICS / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
    lines = ["# Results summary (auto-generated by run_pipeline.py)\n"]
    a = results.get("track_A", {})
    if a:
        lines += [f"\n## Track A — best model: **{a['best_model']}**",
                  f"- Test accuracy: {a['test_metrics']['accuracy']:.4f}",
                  f"- Macro F1: {a['test_metrics']['f1_macro']:.4f}",
                  f"- Macro ROC-AUC (OvR): {a['macro_roc_auc']:.4f}"]
    b = results.get("track_B_models", {})
    if b:
        lines.append("\n## Track B — test metrics")
        lines.append("| model | R² | RMSE | MAE |\n|---|---|---|---|")
        for k, v in b.items():
            lines.append(f"| {k} | {v['r2']:.3f} | {v['rmse']:.3f} | {v['mae']:.3f} |")
    (C.OUTPUTS / "results_summary.md").write_text("\n".join(lines))
    print("\nSaved outputs/metrics/results.json and outputs/results_summary.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="smaller yield subsample")
    args = ap.parse_args()
    t = time.time()
    results = {}
    clf, Xa, _ = run_track_A(results)
    reg, Xb, _ = run_track_B(results, fast=args.fast)
    run_climate(results, clf, Xa, reg, Xb)
    write_summary(results)
    print(f"\nDone in {time.time() - t:.0f}s.")


if __name__ == "__main__":
    main()
