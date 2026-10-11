"""Turn a risk estimate into a readable next-step recommendation.

Combines Project 1's idea (route the patient to the right specialist) with Project 2's
(ranked, explained recommendations with guideline context and a disclaimer).

Rules here are deliberately simple and transparent. They are an educational illustration
of how a risk model can feed a triage workflow - not clinical guidance.
"""
from __future__ import annotations

DISCLAIMER = ("Educational decision-support demo trained on 1980s research data from 4 hospitals. "
              "It is not a diagnosis and must not replace assessment by a qualified clinician.")

RED_FLAGS = ("Chest pain at rest, pain lasting more than a few minutes, or pain with breathlessness, "
             "sweating, nausea or fainting needs emergency care. Call your local emergency number.")

TIERS = {
    "low": {
        "headline": "Low estimated risk",
        "referral": "Primary care doctor (GP)",
        "urgency": "At the next routine visit",
        "next_steps": [
            "Routine follow-up with primary care.",
            "Keep managing modifiable risk factors (blood pressure, cholesterol, smoking, activity).",
            "Reassess if symptoms appear, change or worsen.",
        ],
    },
    "intermediate": {
        "headline": "Intermediate estimated risk",
        "referral": "Cardiologist, outpatient referral",
        "urgency": "Within a few weeks",
        "next_steps": [
            "Discuss the result with a doctor; an outpatient cardiology referral is reasonable.",
            "Non-invasive testing (e.g. exercise stress test or stress imaging) may be considered by the cardiologist.",
            "Review the risk factors flagged below.",
        ],
    },
    "high": {
        "headline": "High estimated risk",
        "referral": "Cardiologist, prompt review",
        "urgency": "Within days",
        "next_steps": [
            "Arrange a prompt cardiology review.",
            "Further evaluation (stress imaging or coronary angiography) is decided by the cardiologist.",
            "Seek emergency care immediately if red-flag symptoms occur.",
        ],
    },
}


def risk_factor_notes(row: dict) -> list[str]:
    """Plain-language flags from the raw inputs (independent of the model)."""
    notes = []
    bp, chol = row.get("trestbps"), row.get("chol")
    if bp is not None and bp >= 140:
        notes.append(f"Resting blood pressure {bp:.0f} mmHg is in the hypertensive range (>= 140).")
    if chol is not None and chol >= 240:
        notes.append(f"Total cholesterol {chol:.0f} mg/dl is high (>= 240).")
    if row.get("fbs") == 1:
        notes.append("Fasting blood sugar is above 120 mg/dl; screening for diabetes is advisable.")
    if row.get("exang") == 1:
        notes.append("Angina brought on by exercise was reported.")
    oldpeak = row.get("oldpeak")
    if oldpeak is not None and oldpeak >= 2:
        notes.append(f"ST depression of {oldpeak:.1f} mm on exercise is marked (>= 2 mm).")
    return notes


def recommend(tier: str, row: dict) -> dict:
    t = TIERS[tier]
    return {"tier": tier, **t, "risk_factor_notes": risk_factor_notes(row),
            "red_flags": RED_FLAGS, "disclaimer": DISCLAIMER}
