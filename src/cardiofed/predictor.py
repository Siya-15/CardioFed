"""Prediction service: API-vocabulary patient dicts in, full explained assessments out.

Used by the API and usable directly:
    from cardiofed.predictor import Predictor
    Predictor().assess({"age": 58, "sex": "male", "cp": "asymptomatic", "exang": "yes"})
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cardiofed import config as C
from cardiofed.features import CATEGORICAL_FEATURES, FEATURES, api_to_row, build_features
from cardiofed.model import CardioModel
from cardiofed.triage import recommend

INPUT_FIELDS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope"]
REQUIRED_FIELDS = ["age", "sex", "cp"]
CODE_TO_NAME = {col: {v: k for k, v in m.items()} for col, m in C.CATEGORY_CODES.items()}
UNITS = {"age": "years", "trestbps": "mmHg", "chol": "mg/dl", "thalach": "bpm", "oldpeak": "mm",
         "hr_reserve_pct": "%"}


class Predictor:
    def __init__(self, model_dir: Path | None = None):
        self.model = CardioModel.load(model_dir or C.MODEL_DIR)
        self.meta = self.model.meta
        self.ranges = self.meta.get("training_ranges", {})

    # ------------------------------------------------------------------ public
    def assess(self, patient: dict) -> dict:
        return self.assess_many([patient])[0]

    def assess_many(self, patients: list[dict]) -> list[dict]:
        rows = [api_to_row({k: p.get(k) for k in INPUT_FIELDS}) for p in patients]
        df = pd.DataFrame(rows, columns=INPUT_FIELDS)
        X = build_features(df)
        probs = self.model.predict_presence(X)
        sev = self.model.predict_severity(X)
        contrib = self.model.explain(X)
        return [self._one(rows[i], X.iloc[i], float(probs[i]), None if sev is None else sev[i], contrib[i])
                for i in range(len(rows))]

    # ------------------------------------------------------------------ internals
    def _one(self, row: dict, x: pd.Series, p: float, sev, contrib: np.ndarray) -> dict:
        tier = self.model.tier(p)
        missing = [f for f in INPUT_FIELDS if row.get(f) is None]
        return {
            "probability": round(p, 4),
            "risk_percent": round(100 * p, 1),
            "tier": tier,
            "thresholds": {k: round(v, 4) for k, v in self.model.thresholds.items()},
            "recommendation": recommend(tier, row),
            "drivers": self._drivers(row, x, contrib),
            "severity_if_present": self._severity(sev),
            "missing_fields": missing,
            "warnings": self._warnings(row, missing),
            "model_version": self.meta.get("version"),
        }

    def _drivers(self, row: dict, x: pd.Series, contrib: np.ndarray, top: int = 6) -> list[dict]:
        order = np.argsort(-np.abs(contrib[:-1]))[:top]
        out = []
        for i in order:
            f = FEATURES[i]
            out.append({
                "feature": f,
                "label": C.FEATURE_LABELS[f],
                "value": self._display(f, row, x),
                "contribution": round(float(contrib[i]), 4),  # log-odds; > 0 raises risk
                "direction": "raises risk" if contrib[i] > 0 else "lowers risk",
            })
        return out

    def _display(self, f: str, row: dict, x: pd.Series) -> str:
        if f == "hr_reserve_pct":
            v = x[f]
            return "not provided" if pd.isna(v) else f"{v:.0f}%"
        v = row.get(f)
        if v is None:
            return "not provided"
        if f in CODE_TO_NAME:
            return CODE_TO_NAME[f][int(v)].replace("_", " ")
        return f"{v:g} {UNITS.get(f, '')}".strip()

    def _severity(self, sev) -> dict | None:
        if sev is None:
            return None
        classes = self.meta.get("severity_classes") or []
        m = self.meta.get("metrics", {}).get("severity", {}).get("xgboost", {})
        return {"probabilities": {c: round(float(s), 4) for c, s in zip(classes, sev)},
                "most_likely": classes[int(np.argmax(sev))],
                "note": ("Experimental: estimated angiographic grade assuming disease is present. "
                         f"Test macro-F1 {m.get('macro_f1', float('nan')):.2f} on {m.get('n', '?')} patients.")}

    def _warnings(self, row: dict, missing: list[str]) -> list[str]:
        w = []
        for f in FEATURES:
            if f in CATEGORICAL_FEATURES or f not in self.ranges or row.get(f) is None:
                continue
            lo, hi = self.ranges[f]
            if not lo <= row[f] <= hi:
                w.append(f"{C.FEATURE_LABELS[f]} = {row[f]:g} is outside the training data range "
                         f"({lo:g}-{hi:g}); the estimate is less reliable.")
        if len(missing) >= 5:
            w.append(f"{len(missing)} inputs were not provided; the estimate relies mostly on the remaining fields.")
        return w
