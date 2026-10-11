"""Shared evaluation helpers: splits, metrics with bootstrap CIs, thresholds."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split

from cardiofed import config as C


def strata(df: pd.DataFrame) -> pd.Series:
    """Stratify on hospital x outcome so every split keeps each site's case mix."""
    return df["hospital"] + "_" + df[C.TARGET].astype(str)


def train_test_indices(df: pd.DataFrame, test_size: float = 0.2):
    """The one train/test split used everywhere (central models, baselines, federated sim)."""
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=test_size, stratify=strata(df), random_state=C.RANDOM_STATE)
    return np.sort(tr), np.sort(te)


def cv_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = C.RANDOM_STATE):
    return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(df, strata(df)))


def binary_metrics(y, p, threshold: float = 0.5) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else None,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "accuracy": float(accuracy_score(y, pred)),
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "threshold": float(threshold),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def bootstrap_ci(y, p, metric=roc_auc_score, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Percentile 95% CI by resampling patients."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        stats.append(metric(y[i], p[i]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def threshold_for_sensitivity(y, p, target: float = 0.90) -> float:
    """Highest threshold that still catches >= target of the diseased patients."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = thr[tpr >= target]
    return float(min(ok.max(), 1.0))


def threshold_for_specificity(y, p, target: float = 0.90) -> float:
    """Lowest threshold that keeps >= target of the healthy patients below it."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = thr[(1 - fpr) >= target]
    return float(min(ok.min(), 1.0))


def multiclass_metrics(y, proba) -> dict:
    pred = np.asarray(proba).argmax(axis=1)
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "n": int(len(y))}


def r3(d):
    """Round floats in nested dicts/lists for readable JSON."""
    if isinstance(d, dict):
        return {k: r3(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [r3(v) for v in d]
    if isinstance(d, (float, np.floating)):
        return round(float(d), 4)
    if isinstance(d, np.integer):
        return int(d)
    return d
