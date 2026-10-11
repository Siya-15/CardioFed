import type { PatientInput } from "./types";

// Form state: every field as a string ("" = not provided), converted to PatientInput on submit.
export type FormValues = Record<keyof PatientInput, string>;

export const EMPTY: FormValues = {
  age: "", sex: "", cp: "", trestbps: "", chol: "", fbs: "", restecg: "", thalach: "", exang: "", oldpeak: "", slope: "",
};

export interface NumberField {
  name: "age" | "trestbps" | "chol" | "thalach" | "oldpeak";
  label: string;
  unit: string;
  min: number;
  max: number;
  step?: number;
  required?: boolean;
}

// Ranges match config.VALID_RANGES on the server, so the browser catches the same errors first.
export const NUMBER_FIELDS: Record<NumberField["name"], NumberField> = {
  age: { name: "age", label: "Age", unit: "years", min: 18, max: 100, required: true },
  trestbps: { name: "trestbps", label: "Blood pressure (systolic)", unit: "mmHg", min: 60, max: 250 },
  chol: { name: "chol", label: "Total cholesterol", unit: "mg/dl", min: 80, max: 700 },
  thalach: { name: "thalach", label: "Maximum heart rate reached", unit: "bpm", min: 50, max: 230 },
  oldpeak: { name: "oldpeak", label: "ST depression vs rest", unit: "mm", min: -3, max: 7, step: 0.1 },
};

export interface Option { value: string; label: string; detail?: string }

export const SEX: Option[] = [{ value: "female", label: "Female" }, { value: "male", label: "Male" }];
export const YES_NO_UNKNOWN: Option[] = [
  { value: "no", label: "No" }, { value: "yes", label: "Yes" }, { value: "", label: "Unknown" },
];
export const CHEST_PAIN: Option[] = [
  { value: "typical_angina", label: "Typical angina", detail: "Pressure-like pain brought on by exertion and relieved by rest" },
  { value: "atypical_angina", label: "Atypical angina", detail: "Has some, but not all, features of typical angina" },
  { value: "non_anginal_pain", label: "Non-anginal pain", detail: "Chest pain that does not look cardiac" },
  { value: "asymptomatic", label: "No chest pain", detail: "Referred for other reasons (in this dataset, a strong risk marker)" },
];
export const REST_ECG: Option[] = [
  { value: "", label: "Unknown" }, { value: "normal", label: "Normal" },
  { value: "st_t_abnormality", label: "ST-T wave abnormality" }, { value: "lv_hypertrophy", label: "Left ventricular hypertrophy" },
];
export const SLOPE: Option[] = [
  { value: "", label: "Unknown" }, { value: "upsloping", label: "Upsloping" },
  { value: "flat", label: "Flat" }, { value: "downsloping", label: "Downsloping" },
];

export const SAMPLES: Record<string, { label: string; values: Partial<FormValues> }> = {
  lower: { label: "lower risk", values: { age: "44", sex: "female", cp: "non_anginal_pain", trestbps: "118", chol: "205",
    fbs: "no", restecg: "normal", thalach: "170", exang: "no", oldpeak: "0", slope: "upsloping" } },
  higher: { label: "higher risk", values: { age: "61", sex: "male", cp: "asymptomatic", trestbps: "148", chol: "262",
    fbs: "yes", restecg: "lv_hypertrophy", thalach: "112", exang: "yes", oldpeak: "2.4", slope: "flat" } },
  partial: { label: "only the basics", values: { age: "57", sex: "male", cp: "atypical_angina" } },
};

export interface Validation {
  patient: PatientInput | null;
  problems: string[];
  invalid: Set<string>;
}

export function validate(v: FormValues): Validation {
  const problems: string[] = [];
  const invalid = new Set<string>();
  const out: Record<string, string | number> = {};

  for (const f of Object.values(NUMBER_FIELDS)) {
    const raw = v[f.name].trim();
    if (raw === "") continue;
    const n = Number(raw);
    if (!Number.isFinite(n) || n < f.min || n > f.max) {
      invalid.add(f.name);
      problems.push(`${f.label} must be between ${f.min} and ${f.max}`);
      continue;
    }
    out[f.name] = f.name === "age" ? Math.round(n) : n;
  }
  for (const name of ["sex", "cp", "fbs", "restecg", "exang", "slope"] as const) {
    if (v[name]) out[name] = v[name];
  }
  if (out.age === undefined && !invalid.has("age")) { invalid.add("age"); problems.push("Enter the patient's age"); }
  if (!out.sex) problems.push("Choose the patient's sex");
  if (!out.cp) problems.push("Choose the chest pain type");

  return { patient: problems.length ? null : (out as unknown as PatientInput), problems, invalid };
}
