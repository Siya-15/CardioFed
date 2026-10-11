"""Single source of truth for paths, column definitions and category codes.

Everything that both training and serving need lives here, so the two can never drift apart.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("CARDIOFED_ROOT", Path(__file__).resolve().parents[2]))
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = Path(os.getenv("CARDIOFED_MODEL_DIR", PROJECT_ROOT / "models"))
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"

RANDOM_STATE = 42

# ---------------------------------------------------------------- source data
# UCI Heart Disease (Janosi, Steinbrunn, Pfisterer, Detrano, 1988), CC BY 4.0.
# https://archive.ics.uci.edu/dataset/45/heart+disease
UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
HOSPITALS = {
    # file suffix -> (display name, expected rows)
    "cleveland": ("Cleveland Clinic (USA)", 303),
    "hungarian": ("Hungarian Institute of Cardiology (Budapest)", 294),
    "switzerland": ("University Hospital Zurich & Basel (Switzerland)", 123),
    "va": ("VA Medical Center Long Beach (USA)", 200),
}

# The 14 attributes used in the "processed" files, in file order.
RAW_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
               "exang", "oldpeak", "slope", "ca", "thal", "num"]

# ---------------------------------------------------------------- features
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY = ["sex", "fbs", "exang"]
CATEGORICAL = ["cp", "restecg", "slope", "thal"]

# Human-readable values used by the API and UI  <->  UCI integer codes used in the data.
# The API speaks names, the model speaks codes; this table is the only translation layer.
CATEGORY_CODES: dict[str, dict[str, int]] = {
    "sex": {"female": 0, "male": 1},
    "cp": {"typical_angina": 1, "atypical_angina": 2, "non_anginal_pain": 3, "asymptomatic": 4},
    "fbs": {"no": 0, "yes": 1},
    "restecg": {"normal": 0, "st_t_abnormality": 1, "lv_hypertrophy": 2},
    "exang": {"no": 0, "yes": 1},
    "slope": {"upsloping": 1, "flat": 2, "downsloping": 3},
    "thal": {"normal": 3, "fixed_defect": 6, "reversible_defect": 7},
}

FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex (male)",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Serum cholesterol",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG",
    "thalach": "Max heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (exercise vs rest)",
    "slope": "Slope of peak exercise ST segment",
    "ca": "Major vessels coloured by fluoroscopy",
    "thal": "Thallium stress test",
    "hr_reserve_pct": "Max heart rate as % of age-predicted max",
}

# Physiologically possible ranges. Values outside are recording errors -> missing.
# (Wide on purpose: we only remove the impossible, not the unusual.)
VALID_RANGES = {
    "age": (18, 100),
    "trestbps": (60, 250),     # mmHg; 0 appears in the raw data as a missing-value code
    "chol": (80, 700),         # mg/dl; 0 is used as a missing-value code in Switzerland/VA
    "thalach": (50, 230),      # bpm
    "oldpeak": (-3.0, 7.0),    # mm; negative = ST elevation, rare but real
    "ca": (0, 3),
}

TARGET = "has_disease"   # num > 0  (angiographic narrowing > 50% in any major vessel)
SEVERITY = "num"         # 0..4 in the source data
