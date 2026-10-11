"""Model wrapper used by training, evaluation and the API.

Two XGBoost models (the Project-2 pattern, applied to real data):
    presence  - P(heart disease)                          binary classifier, all 4 hospitals
    severity  - grade 1 / 2 / 3-4 given disease is present  3-class classifier, graded sites only

Both are wrapped with stochastic imputation (see imputation.py). Presence probabilities are
Platt-calibrated on out-of-fold predictions, and triage thresholds are chosen on those same
out-of-fold predictions, never on the test set.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from cardiofed.features import FEATURES
from cardiofed.imputation import StochasticImputer

SEVERITY_CLASSES = ["grade 1", "grade 2", "grade 3-4"]
MC_DRAWS = 100  # imputation draws averaged at inference when inputs are missing (~1 pt MC noise)


class BoosterModel:
    """Version-proof predictor around a raw xgboost Booster.

    The scikit-learn wrapper's save/load format changes between xgboost/sklearn releases;
    the Booster JSON format does not. Training uses XGBClassifier, serving uses this."""

    def __init__(self, booster: xgb.Booster):
        self.booster = booster

    @classmethod
    def from_classifier(cls, clf) -> "BoosterModel":
        return clf if isinstance(clf, cls) else cls(clf.get_booster())

    @classmethod
    def load(cls, path) -> "BoosterModel":
        b = xgb.Booster()
        b.load_model(str(path))
        return cls(b)

    def save(self, path) -> None:
        self.booster.save_model(str(path))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.booster.predict(xgb.DMatrix(X, enable_categorical=True))
        return np.column_stack([1 - p, p]) if p.ndim == 1 else p

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)


def severity_label(num: pd.Series) -> np.ndarray:
    """UCI num 1..4 -> 0 (grade 1), 1 (grade 2), 2 (grade 3-4)."""
    return np.clip(num.astype(int).to_numpy() - 1, 0, 2)


def _stacked(X: pd.DataFrame, imputer: StochasticImputer, k: int) -> tuple[pd.DataFrame, int]:
    """All k imputed copies of X stacked into one frame, so the model is called once."""
    cols = [c for c in imputer.columns if c in X.columns]
    if not X[cols].isna().any().any():
        return X, 1
    rep = pd.concat([X] * k, ignore_index=True)
    return imputer.fill(rep, np.random.default_rng(1)), k


def mc_proba(model, X: pd.DataFrame, imputer: StochasticImputer, k: int = MC_DRAWS) -> np.ndarray:
    """Average predict_proba over imputation draws."""
    Xs, n = _stacked(X, imputer, k)
    p = model.predict_proba(Xs)
    return p.reshape(n, len(X), -1).mean(axis=0)


def fit_xgb(params: dict, X: pd.DataFrame, y: np.ndarray, imputer: StochasticImputer, seed: int = 0,
            sample_weight=None):
    """Impute once with random draws, then fit."""
    Xf = imputer.fill(X, np.random.default_rng(seed))
    return xgb.XGBClassifier(**params).fit(Xf, y, sample_weight=sample_weight)


@dataclass
class Calibrator:
    """Platt scaling on the logit: p = sigmoid(a * logit(raw) + b)."""
    a: float = 1.0
    b: float = 0.0

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "Calibrator":
        from sklearn.linear_model import LogisticRegression
        z = _logit(raw).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * _logit(raw) + self.b)))


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class CardioModel:
    presence: BoosterModel
    severity: BoosterModel | None
    imputer: StochasticImputer
    calibrator: Calibrator
    thresholds: dict            # {"low": t_sens90, "high": t_spec90}
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.presence = BoosterModel.from_classifier(self.presence)
        if self.severity is not None:
            self.severity = BoosterModel.from_classifier(self.severity)

    # ------------------------------------------------------------ predict
    def predict_presence(self, X: pd.DataFrame) -> np.ndarray:
        raw = mc_proba(self.presence, X[FEATURES], self.imputer)[:, 1]
        return self.calibrator(raw)

    def predict_severity(self, X: pd.DataFrame) -> np.ndarray | None:
        if self.severity is None:
            return None
        return mc_proba(self.severity, X[FEATURES], self.imputer)

    def tier(self, p: float) -> str:
        if p < self.thresholds["low"]:
            return "low"
        if p >= self.thresholds["high"]:
            return "high"
        return "intermediate"

    def explain(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP contributions (log-odds) of the presence model, averaged over imputation draws.
        Returns (n_rows, n_features + 1); last column is the bias term."""
        Xs, n = _stacked(X[FEATURES], self.imputer, MC_DRAWS)
        c = self.presence.contributions(Xs)
        return c.reshape(n, len(X), -1).mean(axis=0)

    # ------------------------------------------------------------ persistence
    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.presence.save(out_dir / "presence_model.json")
        if self.severity is not None:
            self.severity.save(out_dir / "severity_model.json")
        meta = {**self.meta,
                "features": FEATURES,
                "imputer": self.imputer.to_dict(),
                "calibration": {"a": self.calibrator.a, "b": self.calibrator.b},
                "thresholds": self.thresholds,
                "severity_classes": SEVERITY_CLASSES if self.severity is not None else None}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, model_dir: Path) -> "CardioModel":
        model_dir = Path(model_dir)
        for name in ("presence_model.json", "meta.json"):
            if not (model_dir / name).is_file():
                raise FileNotFoundError(f"{model_dir / name} not found - train first: python -m cardiofed.train")
        meta = json.loads((model_dir / "meta.json").read_text())
        if meta["features"] != FEATURES:
            raise RuntimeError("Saved model was trained on a different feature list - retrain it.")
        presence = BoosterModel.load(model_dir / "presence_model.json")
        severity = None
        if (model_dir / "severity_model.json").is_file():
            severity = BoosterModel.load(model_dir / "severity_model.json")
        return cls(presence=presence, severity=severity,
                   imputer=StochasticImputer.from_dict(meta["imputer"]),
                   calibrator=Calibrator(**meta["calibration"]),
                   thresholds=meta["thresholds"], meta=meta)
