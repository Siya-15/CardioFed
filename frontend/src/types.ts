// Mirrors the FastAPI response models in src/cardiofed/api/schemas.py.

export type Tier = "low" | "intermediate" | "high";

export interface PatientInput {
  age: number;
  sex: "female" | "male";
  cp: "typical_angina" | "atypical_angina" | "non_anginal_pain" | "asymptomatic";
  trestbps?: number;
  chol?: number;
  fbs?: "yes" | "no";
  restecg?: "normal" | "st_t_abnormality" | "lv_hypertrophy";
  thalach?: number;
  exang?: "yes" | "no";
  oldpeak?: number;
  slope?: "upsloping" | "flat" | "downsloping";
}

export interface Driver {
  feature: string;
  label: string;
  value: string;
  contribution: number; // log-odds; > 0 raises risk
  direction: string;
}

export interface Recommendation {
  tier: Tier;
  headline: string;
  referral: string;
  urgency: string;
  next_steps: string[];
  risk_factor_notes: string[];
  red_flags: string;
  disclaimer: string;
}

export interface Severity {
  probabilities: Record<string, number>;
  most_likely: string;
  note: string;
}

export interface Assessment {
  probability: number;
  risk_percent: number;
  tier: Tier;
  thresholds: { low: number; high: number };
  recommendation: Recommendation;
  drivers: Driver[];
  severity_if_present: Severity | null;
  missing_fields: string[];
  warnings: string[];
  model_version: string | null;
}

export interface BinaryMetrics {
  auc: number | null;
  brier: number;
  accuracy: number;
  auc_ci95?: [number, number];
}

export interface HospitalHoldout {
  n: number;
  prevalence: number;
  xgboost_auc: number;
  logreg_auc: number;
}

export interface FederatedRow {
  hospital_mean_auc: number;
  cleveland: number | null;
  hungarian: number | null;
  switzerland: number | null;
  va: number | null;
}

export interface ModelSummary {
  version: string;
  data: { patients: number; prevalence: number; source: string };
  thresholds: { low: number; high: number };
  hospitals: Record<string, string>;
  metrics: {
    test: {
      models: Record<string, BinaryMetrics>;
      per_hospital: Record<string, { n: number; positives: number; auc: number | null }>;
    };
    leave_one_hospital_out: Record<string, HospitalHoldout>;
  };
  federated: Record<string, FederatedRow> | null;
}
