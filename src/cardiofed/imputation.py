"""Stochastic imputation: the fix for the 'missing cholesterol = heart disease' shortcut.

Why not let XGBoost handle NaN natively?
    Switzerland recorded no cholesterol at all and 94% of its patients have disease. With
    native NaN handling the model learns "chol missing -> high risk": leaving the field blank
    in the UI raised predicted risk by ~11 points (see reports/feature_selection.json).

Why not median imputation?
    Every imputed patient then sits on the exact same value, and trees can still isolate it.

What we do instead:
    Training  - each missing value is replaced by a random draw from that column's observed
                training values, so "was missing" carries no signal.
    Inference - missing inputs are filled K times with random draws and the K predictions are
                averaged, i.e. the risk is marginalised over what the unknown value could be.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

IMPUTED_COLUMNS = ["trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope"]


def cat_to_float(s: pd.Series) -> np.ndarray:
    """Categorical -> float array with NaN for missing (avoids pandas' NaN-cast warning)."""
    if not isinstance(s.dtype, pd.CategoricalDtype):
        return s.to_numpy(dtype=float, na_value=np.nan).copy()
    cats = np.asarray(s.cat.categories, dtype=float)
    codes = s.cat.codes.to_numpy()
    return np.where(codes >= 0, cats[np.clip(codes, 0, None)], np.nan)


class StochasticImputer:
    def __init__(self, columns: list[str] | None = None, pool_size: int = 400, seed: int = 0):
        self.columns = columns or IMPUTED_COLUMNS
        self.pool_size = pool_size
        self.seed = seed
        self.pools: dict[str, list[float]] = {}

    # -------- fit / (de)serialise
    def fit(self, X: pd.DataFrame) -> "StochasticImputer":
        rng = np.random.default_rng(self.seed)
        for c in self.columns:
            vals = X[c].dropna().astype(float).to_numpy()
            if len(vals) > self.pool_size:  # a representative sample keeps meta.json small
                vals = rng.choice(vals, self.pool_size, replace=False)
            self.pools[c] = [round(float(v), 3) for v in vals]
        return self

    def to_dict(self) -> dict:
        return {"columns": self.columns, "pools": self.pools}

    @classmethod
    def from_dict(cls, d: dict) -> "StochasticImputer":
        obj = cls(columns=d["columns"])
        obj.pools = d["pools"]
        return obj

    # -------- transform
    def fill(self, X: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        X = X.copy()
        for c in self.columns:
            if c not in X.columns:
                continue
            miss = X[c].isna().to_numpy()
            if not miss.any() or not self.pools.get(c):
                continue  # nothing to fill, or column never observed (e.g. chol at Zurich)
            draws = rng.choice(np.asarray(self.pools[c]), miss.sum())
            if isinstance(X[c].dtype, pd.CategoricalDtype):
                cats = X[c].cat.categories
                s = cat_to_float(X[c])
                s[miss] = draws
                X[c] = pd.Categorical(s, categories=cats)
            else:
                X.loc[miss, c] = draws
        if "hr_reserve_pct" in X.columns:  # derived feature must follow its imputed input
            X["hr_reserve_pct"] = X["thalach"] / (220.0 - X["age"]) * 100.0
        return X

    def draws(self, X: pd.DataFrame, k: int, seed: int = 1):
        """Yield k imputed copies of X (identical copies if nothing is missing)."""
        rng = np.random.default_rng(seed)
        if not X[[c for c in self.columns if c in X.columns]].isna().any().any():
            yield X
            return
        for _ in range(k):
            yield self.fill(X, rng)
