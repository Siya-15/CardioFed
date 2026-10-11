"""Request/response schemas. Category names and numeric ranges come from config, so the API
can never accept a value the model was not built for (and a typo'd field is a 422, not silently ignored)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from cardiofed import config as C

R = C.VALID_RANGES


def _lit(col):
    return Literal[tuple(C.CATEGORY_CODES[col])]  # type: ignore[valid-type]


Sex = _lit("sex")
ChestPain = _lit("cp")
YesNo = Literal["yes", "no"]
RestECG = _lit("restecg")
Slope = _lit("slope")


class PatientIn(BaseModel):
    """One patient. Only age, sex and chest pain type are required; unknown values may be omitted
    or sent as null and are averaged over (they do not push the risk up or down)."""
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "age": 58, "sex": "male", "cp": "asymptomatic", "trestbps": 140, "chol": 260, "fbs": "no",
        "restecg": "normal", "thalach": 128, "exang": "yes", "oldpeak": 1.8, "slope": "flat"}})

    age: int = Field(..., ge=R["age"][0], le=R["age"][1], description="years")
    sex: Sex
    cp: ChestPain = Field(..., description="chest pain type")
    trestbps: Optional[float] = Field(None, ge=R["trestbps"][0], le=R["trestbps"][1], description="resting systolic BP, mmHg")
    chol: Optional[float] = Field(None, ge=R["chol"][0], le=R["chol"][1], description="serum cholesterol, mg/dl")
    fbs: Optional[YesNo] = Field(None, description="fasting blood sugar > 120 mg/dl")
    restecg: Optional[RestECG] = Field(None, description="resting ECG result")
    thalach: Optional[float] = Field(None, ge=R["thalach"][0], le=R["thalach"][1], description="max heart rate on exercise, bpm")
    exang: Optional[YesNo] = Field(None, description="exercise-induced angina")
    oldpeak: Optional[float] = Field(None, ge=R["oldpeak"][0], le=R["oldpeak"][1], description="exercise ST depression, mm")
    slope: Optional[Slope] = Field(None, description="slope of the peak exercise ST segment")


class BatchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patients: list[PatientIn] = Field(..., min_length=1, max_length=100)


class Driver(BaseModel):
    feature: str
    label: str
    value: str
    contribution: float
    direction: str


class Recommendation(BaseModel):
    tier: str
    headline: str
    referral: str
    urgency: str
    next_steps: list[str]
    risk_factor_notes: list[str]
    red_flags: str
    disclaimer: str


class Severity(BaseModel):
    probabilities: dict[str, float]
    most_likely: str
    note: str


class AssessmentOut(BaseModel):
    probability: float
    risk_percent: float
    tier: Literal["low", "intermediate", "high"]
    thresholds: dict[str, float]
    recommendation: Recommendation
    drivers: list[Driver]
    severity_if_present: Optional[Severity]
    missing_fields: list[str]
    warnings: list[str]
    model_version: Optional[str]


class BatchOut(BaseModel):
    results: list[AssessmentOut]
