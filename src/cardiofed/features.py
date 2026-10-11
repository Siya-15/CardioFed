"""Feature engineering shared by training, the federated simulation and the API.

One function, `build_features`, turns cleaned rows (UCI integer codes) into the model matrix.
The column list and category codes come from config, so train and serve can't disagree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cardiofed import config as C

# Final feature set (see reports/feature_selection.json for the experiment behind it):
#   - ca and thal are dropped: measured almost only at Cleveland (99% / 90% missing elsewhere),
#     so their *missingness* identifies the hospital rather than the patient.
#   - slope is kept: measured at 3 of 4 sites and clinically central to stress-test reading.
BASE_FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
                 "exang", "oldpeak", "slope"]
DERIVED_FEATURES = ["hr_reserve_pct"]
FEATURES = BASE_FEATURES + DERIVED_FEATURES
CATEGORICAL_FEATURES = [c for c in C.CATEGORICAL if c in FEATURES]


def build_features(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Rows with UCI-coded columns -> model matrix (float numerics, fixed-category categoricals).

    Missing inputs stay NaN; XGBoost learns a default branch direction for them.
    Unknown category codes become NaN instead of raising.
    """
    features = features or FEATURES
    X = df.reindex(columns=[c for c in C.RAW_COLUMNS if c not in ("num",)]).copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)

    # Derived: achieved max heart rate as a share of the age-predicted maximum (220 - age).
    # A low value (chronotropic incompetence) is a known marker of coronary disease.
    X["hr_reserve_pct"] = X["thalach"] / (220.0 - X["age"]) * 100.0

    X = X[features]
    for col in [c for c in C.CATEGORICAL if c in features]:
        codes = sorted(C.CATEGORY_CODES[col].values())
        X[col] = pd.Categorical(X[col].where(X[col].isin(codes)), categories=codes)
    return X


def api_to_row(payload: dict) -> dict:
    """Translate API vocabulary ('male', 'asymptomatic', ...) into UCI codes. None stays None."""
    row = {}
    for key, value in payload.items():
        if key in C.CATEGORY_CODES and value is not None:
            row[key] = C.CATEGORY_CODES[key][value]
        else:
            row[key] = value
    return row


def lr_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot view of the feature matrix for linear models (baseline + federated LR)."""
    out = X.copy()
    cats = [c for c in out.columns if isinstance(out[c].dtype, pd.CategoricalDtype)]
    out = pd.get_dummies(out, columns=cats, dummy_na=False, dtype=float)
    return out.astype(float)


def missing_mask(X: pd.DataFrame) -> np.ndarray:
    return X.isna().to_numpy()
