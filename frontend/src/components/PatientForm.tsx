import { useState, type FormEvent } from "react";
import {
  CHEST_PAIN, EMPTY, NUMBER_FIELDS, REST_ECG, SAMPLES, SEX, SLOPE, YES_NO_UNKNOWN, validate, type FormValues,
} from "../fields";
import type { PatientInput } from "../types";
import { ChoiceList, NumberInput, Segmented, Select } from "./controls";

export function PatientForm({ onSubmit, busy, serverError }: {
  onSubmit: (p: PatientInput) => void; busy: boolean; serverError: string | null;
}) {
  const [values, setValues] = useState<FormValues>(EMPTY);
  const [problems, setProblems] = useState<string[]>([]);
  const [invalid, setInvalid] = useState<Set<string>>(new Set());

  const set = (name: keyof FormValues) => (v: string) => setValues((prev) => ({ ...prev, [name]: v }));
  const num = (name: keyof typeof NUMBER_FIELDS) => (
    <NumberInput field={NUMBER_FIELDS[name]} value={values[name]} invalid={invalid.has(name)} onChange={set(name)} />
  );

  function submit(e: FormEvent) {
    e.preventDefault();
    const r = validate(values);
    setProblems(r.problems);
    setInvalid(r.invalid);
    if (r.patient) onSubmit(r.patient);
  }

  function reset() {
    setValues(EMPTY);
    setProblems([]);
    setInvalid(new Set());
  }

  const error = problems.length ? problems.join(". ") + "." : serverError;

  return (
    <form className="patient-form" noValidate onSubmit={submit} onReset={reset}>
      <div className="form-intro">
        <h1>Estimate the chance of coronary heart disease</h1>
        <p>Enter what you know. Only age, sex and chest pain type are required; anything you leave blank is averaged
          over, so it neither raises nor lowers the estimate.</p>
        <div className="samples">
          <span>Try a sample patient:</span>
          {Object.entries(SAMPLES).map(([key, s]) => (
            <button key={key} type="button" className="link-button"
                    onClick={() => { reset(); setValues({ ...EMPTY, ...s.values }); }}>
              {s.label}
            </button>
          ))}
        </div>
      </div>

      <fieldset>
        <legend>Patient</legend>
        <div className="row">
          {num("age")}
          <Segmented name="sex" label="Sex" required options={SEX} value={values.sex} onChange={set("sex")} />
        </div>
      </fieldset>

      <fieldset>
        <legend>Chest pain <span className="req">required</span></legend>
        <ChoiceList name="cp" label="Chest pain type" options={CHEST_PAIN} value={values.cp} onChange={set("cp")} />
      </fieldset>

      <fieldset>
        <legend>At rest</legend>
        <div className="row">{num("trestbps")}{num("chol")}</div>
        <div className="row">
          <Segmented name="fbs" label="Fasting blood sugar above 120 mg/dl" options={YES_NO_UNKNOWN}
                     value={values.fbs} onChange={set("fbs")} />
          <Select name="restecg" label="Resting ECG" options={REST_ECG} value={values.restecg} onChange={set("restecg")} />
        </div>
      </fieldset>

      <fieldset>
        <legend>Exercise stress test</legend>
        <p className="hint">Leave these blank if no stress test was done.</p>
        <div className="row">{num("thalach")}{num("oldpeak")}</div>
        <div className="row">
          <Segmented name="exang" label="Chest pain during exercise" options={YES_NO_UNKNOWN}
                     value={values.exang} onChange={set("exang")} />
          <Select name="slope" label="ST segment slope at peak" options={SLOPE} value={values.slope} onChange={set("slope")} />
        </div>
      </fieldset>

      {error && <p className="form-error" role="alert">{error}</p>}
      <div className="form-actions">
        <button type="submit" className="primary" disabled={busy}>{busy ? "Estimating…" : "Estimate risk"}</button>
        <button type="reset" className="quiet">Clear form</button>
      </div>
    </form>
  );
}
