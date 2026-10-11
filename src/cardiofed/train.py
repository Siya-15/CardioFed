"""Train, evaluate and save the heart disease models.

    python -m cardiofed.train            # full run (~3-6 min on a laptop)
    python -m cardiofed.train --fast     # smaller grid + fewer bootstrap samples, for CI/tests

Protocol (all decisions made without touching the test set):
    1. One stratified 80/20 split (hospital x outcome), shared with the federated simulation.
    2. Ablation on the training set: which features, and how to treat missing values.
    3. Hyper-parameters by 5-fold CV on the training set.
    4. Platt calibration + triage thresholds from out-of-fold training predictions.
    5. Held-out test metrics with bootstrap 95% CIs, against 3 baselines.
    6. External validation: leave-one-hospital-out (train on 3 hospitals, test on the 4th).
    7. Severity model (graded hospitals only) with its own baselines.
    8. The deployed model is the same recipe refit on all 918 patients.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from cardiofed import __version__
from cardiofed import config as C
from cardiofed.data import load_clean
from cardiofed.evaluation import (binary_metrics, bootstrap_ci, cv_folds, multiclass_metrics, r3,
                                  threshold_for_sensitivity, threshold_for_specificity,
                                  train_test_indices)
from cardiofed.features import BASE_FEATURES, CATEGORICAL_FEATURES, FEATURES, build_features
from cardiofed.imputation import StochasticImputer, cat_to_float
from cardiofed.model import (SEVERITY_CLASSES, Calibrator, CardioModel, fit_xgb, mc_proba,
                             severity_label)

BASE_PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                   enable_categorical=True, subsample=0.8, colsample_bytree=0.8,
                   min_child_weight=3, reg_lambda=1.0, learning_rate=0.03, n_jobs=-1,
                   random_state=C.RANDOM_STATE)
GRID = {"max_depth": [2, 3, 4], "n_estimators": [150, 300, 500]}
FAST_GRID = {"max_depth": [3], "n_estimators": [300]}
CV_DRAWS = 10   # imputation draws during CV (speed); deployment uses model.MC_DRAWS


# ====================================================================== baselines
class Winsoriser(BaseEstimator, TransformerMixin):
    """Clip each numeric column to its 1st-99th percentile. Bounds are learned on the training
    fold only, so test-set outliers never influence them."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_, self.hi_ = np.nanpercentile(X, 1, axis=0), np.nanpercentile(X, 99, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def make_logreg():
    """Linear baseline: median impute -> winsorise outliers -> scale; one-hot categoricals."""
    num = [c for c in FEATURES if c not in CATEGORICAL_FEATURES]
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("clip", Winsoriser()),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("float", FunctionTransformer(as_float)), ("prep", prep),
                     ("clf", LogisticRegression(max_iter=2000, C=1.0))])


def make_random_forest():
    """Project 1's Random Forest configuration, kept as a baseline."""
    rf = RandomForestClassifier(n_estimators=120, max_depth=20, min_samples_leaf=2, max_features="sqrt",
                                class_weight="balanced_subsample", random_state=C.RANDOM_STATE, n_jobs=-1)
    return Pipeline([("float", FunctionTransformer(as_float)), ("imp", SimpleImputer(strategy="median")),
                     ("rf", rf)])


def as_float(X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: cat_to_float(X[c]) for c in X.columns})


# ====================================================================== xgboost helpers
def xgb_cv_oof(X, y, folds, params, draws=CV_DRAWS, stochastic=True):
    oof = np.zeros(len(y))
    for k, (tr, va) in enumerate(folds):
        imp = StochasticImputer().fit(X.iloc[tr])
        if stochastic:
            m = fit_xgb(params, X.iloc[tr], y[tr], imp, seed=k)
            oof[va] = mc_proba(m, X.iloc[va], imp, draws)[:, 1]
        else:
            m = xgb.XGBClassifier(**params).fit(X.iloc[tr], y[tr])
            oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


def blank_shift(model_predict, X_complete: pd.DataFrame, cols=("chol", "fbs", "trestbps", "oldpeak")):
    """How much does leaving one field blank move predicted risk, on patients who had it?"""
    base = model_predict(X_complete)
    out = {}
    for c in cols:
        Xb = X_complete.copy()
        Xb[c] = np.nan if not isinstance(Xb[c].dtype, pd.CategoricalDtype) else \
            pd.Categorical([np.nan] * len(Xb), categories=Xb[c].cat.categories)
        out[c] = float(np.mean(model_predict(Xb) - base))
    return out


# ====================================================================== steps
def ablation(df, tr_idx, params, folds_tr):
    """Feature set x missing-value strategy, evaluated on the training set only."""
    print("\n[ablation] feature set and missing-value handling (training set only)")
    dtr = df.iloc[tr_idx].reset_index(drop=True)
    y = dtr[C.TARGET].to_numpy()
    site = dtr["hospital"].to_numpy()
    variants = {
        "all 13 features, native NaN": (BASE_FEATURES + ["ca", "thal", "hr_reserve_pct"], False),
        "drop ca/thal, native NaN": (FEATURES, False),
        "drop ca/thal, stochastic imputation (chosen)": (FEATURES, True),
    }
    complete = (site == "cleveland") | (site == "hungarian")
    results = {}
    for name, (feats, stochastic) in variants.items():
        X = build_features(dtr, feats)
        cv_auc = roc_auc_score(y, xgb_cv_oof(X, y, folds_tr, params, stochastic=stochastic))
        loho = {}
        for s in C.HOSPITALS:
            a, b = site != s, site == s
            imp = StochasticImputer(columns=[c for c in StochasticImputer().columns if c in feats]).fit(X[a])
            m = fit_xgb(params, X[a], y[a], imp) if stochastic else xgb.XGBClassifier(**params).fit(X[a], y[a])
            p = mc_proba(m, X[b], imp, CV_DRAWS)[:, 1] if stochastic else m.predict_proba(X[b])[:, 1]
            loho[s] = float(roc_auc_score(y[b], p))
        imp = StochasticImputer(columns=[c for c in StochasticImputer().columns if c in feats]).fit(X)
        full = fit_xgb(params, X, y, imp) if stochastic else xgb.XGBClassifier(**params).fit(X, y)
        pred = (lambda Z: mc_proba(full, Z, imp, CV_DRAWS)[:, 1]) if stochastic else \
            (lambda Z: full.predict_proba(Z)[:, 1])
        miss = X.isna().astype(int)
        site_acc = float(np.mean([DecisionTreeClassifier(max_depth=4, random_state=0)
                                  .fit(miss.iloc[a_], site[a_]).score(miss.iloc[b_], site[b_])
                                  for a_, b_ in folds_tr]))
        results[name] = {"cv_auc": cv_auc, "loho_auc": loho, "loho_mean": float(np.mean(list(loho.values()))),
                         "risk_shift_if_blank": blank_shift(pred, X[complete]),
                         "hospital_predictable_from_missingness": site_acc}
        r = results[name]
        print(f"  {name:46s} CV AUC {r['cv_auc']:.3f} | LOHO mean {r['loho_mean']:.3f} | "
              f"blank chol shifts risk {r['risk_shift_if_blank']['chol']:+.3f} | "
              f"hospital-from-missingness acc {site_acc:.2f}")
    return results


def tune(X, y, folds, grid):
    print("\n[tune] 5-fold CV on the training set")
    best, rows = None, []
    for depth, n in itertools.product(grid["max_depth"], grid["n_estimators"]):
        params = {**BASE_PARAMS, "max_depth": depth, "n_estimators": n}
        auc = roc_auc_score(y, xgb_cv_oof(X, y, folds, params))
        rows.append({"max_depth": depth, "n_estimators": n, "cv_auc": auc})
        print(f"  depth {depth} trees {n:4d}: CV AUC {auc:.4f}")
        # prefer simpler models unless the gain is meaningful (> 0.002 AUC)
        if best is None or auc > best[0] + 0.002:
            best = (auc, params)
    print(f"  -> chosen: depth {best[1]['max_depth']}, trees {best[1]['n_estimators']}")
    return best[1], rows


def evaluate_test(Xtr, ytr, Xte, yte, site_te, params, folds_tr, n_boot):
    """Fit on train; calibrate/threshold on train OOF; score the held-out test set."""
    oof_raw = xgb_cv_oof(Xtr, ytr, folds_tr, params)
    cal = Calibrator().fit(oof_raw, ytr)
    oof = cal(oof_raw)
    thr = {"low": threshold_for_sensitivity(ytr, oof, 0.90), "high": threshold_for_specificity(ytr, oof, 0.90)}

    imp = StochasticImputer().fit(Xtr)
    m = fit_xgb(params, Xtr, ytr, imp)
    p_xgb = cal(mc_proba(m, Xte, imp)[:, 1])

    lr = make_logreg().fit(Xtr, ytr)
    rf = make_random_forest().fit(Xtr, ytr)
    preds = {
        "majority class": np.full(len(yte), ytr.mean()),
        "logistic regression": lr.predict_proba(Xte)[:, 1],
        "random forest (Project 1 config)": rf.predict_proba(Xte)[:, 1],
        "xgboost (CardioFed)": p_xgb,
    }
    print(f"\n[test] held-out test set: {len(yte)} patients ({int(yte.sum())} with disease)")
    results = {}
    for name, p in preds.items():
        met = binary_metrics(yte, p, 0.5)
        if name != "majority class":
            met["auc_ci95"] = bootstrap_ci(yte, p, n=n_boot)
        results[name] = met
        ci = met.get("auc_ci95")
        print(f"  {name:34s} AUC {met['auc'] or 0.5:.3f}"
              + (f" [{ci[0]:.3f}-{ci[1]:.3f}]" if ci else "") + f" | Brier {met['brier']:.3f}")

    screening = binary_metrics(yte, p_xgb, thr["low"])
    rule_in = binary_metrics(yte, p_xgb, thr["high"])
    print(f"  screening threshold {thr['low']:.2f}: sensitivity {screening['sensitivity']:.2f}, "
          f"specificity {screening['specificity']:.2f}")
    print(f"  high-risk threshold {thr['high']:.2f}: sensitivity {rule_in['sensitivity']:.2f}, "
          f"specificity {rule_in['specificity']:.2f}")

    per_site = {}
    for s in C.HOSPITALS:
        mask = site_te == s
        ys = yte[mask]
        enough = min(ys.sum(), (1 - ys).sum()) >= 5
        per_site[s] = {"n": int(mask.sum()), "positives": int(ys.sum()),
                       "auc": float(roc_auc_score(ys, p_xgb[mask])) if enough else None,
                       "note": None if enough else "fewer than 5 patients in one class - AUC not reported"}
    calib_curve = _calibration_bins(yte, p_xgb)
    return {"models": results, "screening": screening, "high_risk": rule_in, "per_hospital": per_site,
            "calibration_bins": calib_curve, "oof_thresholds": thr}, preds


def _calibration_bins(y, p, bins=5):
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    out = []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (p >= lo) & (p <= hi)
        if m.sum():
            out.append({"mean_predicted": float(p[m].mean()), "observed_rate": float(y[m].mean()), "n": int(m.sum())})
    return out


def leave_one_hospital_out(X, y, site, params):
    print("\n[external] leave-one-hospital-out: train on 3 hospitals, test on the unseen 4th")
    out = {}
    for s in C.HOSPITALS:
        a, b = site != s, site == s
        imp = StochasticImputer().fit(X[a])
        p = mc_proba(fit_xgb(params, X[a], y[a], imp), X[b], imp, CV_DRAWS)[:, 1]
        p_lr = make_logreg().fit(X[a], y[a]).predict_proba(X[b])[:, 1]
        out[s] = {"n": int(b.sum()), "prevalence": float(y[b].mean()),
                  "xgboost_auc": float(roc_auc_score(y[b], p)), "logreg_auc": float(roc_auc_score(y[b], p_lr)),
                  "xgboost_brier": float(np.mean((p - y[b]) ** 2))}
        print(f"  held out {s:12s} n={b.sum():3d} prevalence {y[b].mean():.0%} | "
              f"XGBoost AUC {out[s]['xgboost_auc']:.3f} | LogReg AUC {out[s]['logreg_auc']:.3f}")
    return out


SEV_PARAMS = dict(objective="multi:softprob", num_class=3, eval_metric="mlogloss", tree_method="hist",
                  enable_categorical=True, max_depth=2, n_estimators=200, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=3, random_state=C.RANDOM_STATE, n_jobs=-1)


def severity_eval(df, tr_idx, te_idx):
    """Grade 1 / 2 / 3-4 among patients with disease, at hospitals that recorded grades."""
    print("\n[severity] grade 1 / 2 / 3-4 among diseased patients (Cleveland, Switzerland, VA)")
    graded = (df[C.TARGET] == 1) & (df["severity_available"] == 1)
    tr = np.array([i for i in tr_idx if graded.iloc[i]])
    te = np.array([i for i in te_idx if graded.iloc[i]])
    X = build_features(df)
    ys = severity_label(df["num"])
    imp = StochasticImputer().fit(X.iloc[tr])
    m = fit_xgb(SEV_PARAMS, X.iloc[tr], ys[tr], imp)
    p = mc_proba(m, X.iloc[te], imp)
    lr = Pipeline(make_logreg().steps[:-1] + [("clf", LogisticRegression(max_iter=2000))]).fit(X.iloc[tr], ys[tr])
    counts = np.bincount(ys[tr], minlength=3)
    majority = np.tile(np.eye(3)[counts.argmax()], (len(te), 1))
    res = {"classes": SEVERITY_CLASSES, "train_class_counts": counts.tolist(),
           "xgboost": multiclass_metrics(ys[te], p),
           "logistic regression": multiclass_metrics(ys[te], lr.predict_proba(X.iloc[te])),
           "majority class": multiclass_metrics(ys[te], majority)}
    for k in ("majority class", "logistic regression", "xgboost"):
        print(f"  {k:20s} accuracy {res[k]['accuracy']:.3f} | macro F1 {res[k]['macro_f1']:.3f} (n={res[k]['n']})")
    return res


def global_importance(model: CardioModel, X: pd.DataFrame) -> dict:
    contrib = model.explain(X)[:, :-1]
    imp = np.abs(contrib).mean(axis=0)
    order = np.argsort(imp)[::-1]
    return {FEATURES[i]: float(imp[i]) for i in order}


# ====================================================================== figures
def figures(test_preds, yte, test_res, loho, importance, ablation_res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10})
    colors = {"logistic regression": "#8a8a8a", "random forest (Project 1 config)": "#c98b2b",
              "xgboost (CardioFed)": "#c0392b"}

    fig, ax = plt.subplots(figsize=(5, 4.4))
    for name, p in test_preds.items():
        if name == "majority class":
            continue
        fpr, tpr, _ = roc_curve(yte, p)
        ax.plot(fpr, tpr, color=colors[name], lw=2 if "xgboost" in name else 1.4,
                label=f"{name} (AUC {test_res['models'][name]['auc']:.3f})")
    ax.plot([0, 1], [0, 1], ls=":", color="#bbbbbb")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC - held-out test set")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIGURES / "roc_test.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    b = test_res["calibration_bins"]
    ax.plot([0, 1], [0, 1], ls=":", color="#bbbbbb", label="perfect calibration")
    ax.plot([x["mean_predicted"] for x in b], [x["observed_rate"] for x in b], "o-", color="#c0392b",
            label="CardioFed (Platt-calibrated)")
    ax.set(xlabel="Predicted risk", ylabel="Observed disease rate", title="Calibration - test set")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(C.FIGURES / "calibration_test.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    sites = list(loho); x = np.arange(len(sites)); w = 0.38
    ax.bar(x - w / 2, [loho[s]["logreg_auc"] for s in sites], w, color="#8a8a8a", label="Logistic regression")
    ax.bar(x + w / 2, [loho[s]["xgboost_auc"] for s in sites], w, color="#c0392b", label="XGBoost")
    ax.set_xticks(x, [s.capitalize() if s != "va" else "VA Long Beach" for s in sites])
    ax.set(ylim=(0.5, 1.0), ylabel="AUC on unseen hospital", title="External validation (leave-one-hospital-out)")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(C.FIGURES / "loho_auc.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4))
    items = list(importance.items())[::-1]
    ax.barh([C.FEATURE_LABELS.get(k, k) for k, _ in items], [v for _, v in items], color="#c0392b")
    ax.set(xlabel="Mean |SHAP value| (log-odds)", title="What drives the prediction")
    fig.tight_layout(); fig.savefig(C.FIGURES / "shap_importance.png"); plt.close(fig)

    if not ablation_res:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    names = list(ablation_res)
    short = ["All 13 features\nnative NaN", "Drop ca/thal\nnative NaN", "Drop ca/thal\nstochastic imputation"]
    vals = [ablation_res[n]["risk_shift_if_blank"]["chol"] * 100 for n in names]
    ax.bar(short, vals, color=["#8a8a8a", "#8a8a8a", "#c0392b"])
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set(ylabel="Risk change (percentage points)", title="Effect of leaving cholesterol blank",
           ylim=(min(-1, min(vals) - 1), max(vals) + 2))
    for i, v in enumerate(vals):
        ax.text(i, max(v, 0) + 0.4, f"{v:+.1f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(C.FIGURES / "missingness_shortcut.png"); plt.close(fig)


# ====================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="tiny grid, fewer bootstraps, skip ablation")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()

    df = load_clean()
    X = build_features(df)
    y = df[C.TARGET].to_numpy()
    site = df["hospital"].to_numpy()
    tr_idx, te_idx = train_test_indices(df)
    dtr = df.iloc[tr_idx].reset_index(drop=True)
    folds_tr = cv_folds(dtr)
    Xtr, ytr = X.iloc[tr_idx].reset_index(drop=True), y[tr_idx]
    Xte, yte = X.iloc[te_idx].reset_index(drop=True), y[te_idx]
    print(f"[data] {len(df)} patients | train {len(tr_idx)} | test {len(te_idx)} | prevalence {y.mean():.1%}")

    default_params = {**BASE_PARAMS, "max_depth": 3, "n_estimators": 300}
    abl = {} if args.fast else ablation(df, tr_idx, default_params, folds_tr)
    params, grid_rows = tune(Xtr, ytr, folds_tr, FAST_GRID if args.fast else GRID)
    test_res, test_preds = evaluate_test(Xtr, ytr, Xte, yte, site[te_idx], params, folds_tr,
                                         n_boot=200 if args.fast else 1000)
    loho = leave_one_hospital_out(X, y, site, params)
    sev = severity_eval(df, tr_idx, te_idx)

    # ---------------- deploy: same recipe, all data
    print("\n[deploy] refitting on all patients")
    folds_all = cv_folds(df)
    oof_raw = xgb_cv_oof(X, y, folds_all, params)
    cal = Calibrator().fit(oof_raw, y)
    oof = cal(oof_raw)
    thresholds = {"low": threshold_for_sensitivity(y, oof, 0.90), "high": threshold_for_specificity(y, oof, 0.90)}
    imp = StochasticImputer().fit(X)
    presence = fit_xgb(params, X, y, imp)
    graded = ((df[C.TARGET] == 1) & (df["severity_available"] == 1)).to_numpy()
    severity = fit_xgb(SEV_PARAMS, X[graded], severity_label(df.loc[graded, "num"]), imp)

    data_hash = hashlib.sha256(b"".join((C.DATA_RAW / f"processed.{s}.data").read_bytes()
                                        for s in C.HOSPITALS)).hexdigest()[:16]
    meta = {
        "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {"source": "UCI Heart Disease (4 hospitals), CC BY 4.0", "patients": int(len(df)),
                 "sha256_16": data_hash, "prevalence": float(y.mean())},
        "params": {k: v for k, v in params.items() if k != "n_jobs"},
        "software": {"python": platform.python_version(), "xgboost": xgb.__version__,
                     "sklearn": sklearn.__version__, "pandas": pd.__version__},
        "metrics": r3({"test": {k: test_res[k] for k in ("models", "screening", "high_risk", "per_hospital")},
                       "leave_one_hospital_out": loho, "severity": sev}),
        "training_ranges": {c: [float(X[c].min()), float(X[c].max())]
                            for c in FEATURES if c not in CATEGORICAL_FEATURES},
    }
    model = CardioModel(presence=presence, severity=severity, imputer=imp, calibrator=cal,
                        thresholds=thresholds, meta=meta)
    model.save(C.MODEL_DIR)
    importance = global_importance(model, X)
    print(f"  thresholds: low < {thresholds['low']:.3f} <= intermediate < {thresholds['high']:.3f} <= high")
    print(f"  saved to {C.MODEL_DIR}/")

    report = r3({"ablation": abl, "tuning": grid_rows, "chosen_params": meta["params"], "test": test_res,
                 "leave_one_hospital_out": loho, "severity": sev, "deployed_thresholds": thresholds,
                 "global_importance": importance})
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    (C.REPORTS / "metrics.json").write_text(json.dumps(report, indent=2))
    if abl:
        (C.REPORTS / "feature_selection.json").write_text(json.dumps(r3(abl), indent=2))
    if not args.no_figures:
        figures(test_preds, yte, test_res, loho, importance, abl)
    print(f"\n[done] {time.time() - t0:.0f}s - metrics in reports/metrics.json, figures in reports/figures/")


if __name__ == "__main__":
    main()
