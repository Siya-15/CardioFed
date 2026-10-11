import json

import numpy as np
import pytest

from cardiofed import config as C
from cardiofed.model import CardioModel
from tests.conftest import FULL_PATIENT, LOW_PATIENT


def test_outputs_are_valid_and_ordered(predictor):
    hi, lo = predictor.assess(FULL_PATIENT), predictor.assess(LOW_PATIENT)
    for r in (hi, lo):
        assert 0 <= r["probability"] <= 1
        assert r["tier"] in {"low", "intermediate", "high"}
        assert len(r["drivers"]) == 6
        assert abs(sum(r["severity_if_present"]["probabilities"].values()) - 1) < 1e-3
    assert hi["probability"] > lo["probability"]
    assert hi["tier"] == "high" and lo["tier"] == "low"


def test_predictions_are_deterministic(predictor):
    partial = {"age": 57, "sex": "male", "cp": "atypical_angina"}
    assert predictor.assess(partial)["probability"] == predictor.assess(partial)["probability"]


def test_tier_thresholds_are_consistent(predictor):
    t = predictor.model.thresholds
    assert 0 < t["low"] < t["high"] < 1
    assert predictor.model.tier(t["low"] - 1e-6) == "low"
    assert predictor.model.tier(t["high"]) == "high"


@pytest.mark.parametrize("field", ["chol", "fbs", "trestbps", "oldpeak", "thalach"])
def test_leaving_a_field_blank_does_not_push_risk(predictor, field):
    """Regression test for the 'missing cholesterol = disease' shortcut (was +10 points)."""
    import pandas as pd
    from cardiofed.data import load_clean

    df = load_clean()
    rows = df[df.hospital.isin(["cleveland", "hungarian"])].dropna(subset=[field]).sample(150, random_state=0)
    names = {col: {v: k for k, v in m.items()} for col, m in C.CATEGORY_CODES.items()}
    patients = []
    for _, r in rows.iterrows():
        p = {}
        for f in ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope"]:
            v = r[f]
            p[f] = None if pd.isna(v) else (names[f][int(v)] if f in names else float(v))
        p["age"] = int(p["age"])
        patients.append(p)
    base = np.array([x["probability"] for x in predictor.assess_many(patients)])
    blank = np.array([x["probability"] for x in predictor.assess_many([{**p, field: None} for p in patients])])
    assert abs((blank - base).mean()) < 0.03, f"blank {field} shifts risk by {(blank - base).mean():+.3f}"


def test_explanations_add_up_to_the_prediction(predictor):
    import pandas as pd
    from cardiofed.features import api_to_row, build_features

    X = build_features(pd.DataFrame([api_to_row(FULL_PATIENT)]))
    contrib = predictor.model.explain(X)[0]
    raw = predictor.model.presence.predict_proba(X)[0, 1]
    assert np.isclose(1 / (1 + np.exp(-contrib.sum())), raw, atol=1e-4)  # SHAP: sum of parts = logit


def test_model_refuses_mismatched_features(tmp_path):
    for f in ("presence_model.json", "severity_model.json"):
        (tmp_path / f).write_bytes((C.MODEL_DIR / f).read_bytes())
    meta = json.loads((C.MODEL_DIR / "meta.json").read_text())
    meta["features"] = meta["features"][:-1]
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="different feature list"):
        CardioModel.load(tmp_path)


def test_saved_metrics_beat_baseline(predictor):
    m = predictor.meta["metrics"]["test"]["models"]
    assert m["xgboost (CardioFed)"]["auc"] > 0.83
    assert m["xgboost (CardioFed)"]["brier"] < m["majority class"]["brier"]
