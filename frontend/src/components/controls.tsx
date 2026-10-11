import { useId } from "react";
import type { NumberField, Option } from "../fields";

export function NumberInput({ field, value, invalid, onChange }: {
  field: NumberField; value: string; invalid: boolean; onChange: (v: string) => void;
}) {
  return (
    <label className="field">
      <span className="field-label">
        {field.label} {field.required && <span className="req">required</span>}
      </span>
      <span className="input-unit">
        <input
          name={field.name}
          type="number"
          inputMode={field.step ? "decimal" : "numeric"}
          min={field.min}
          max={field.max}
          step={field.step}
          required={field.required}
          placeholder={field.required ? undefined : "not measured"}
          value={value}
          aria-invalid={invalid || undefined}
          onChange={(e) => onChange(e.target.value)}
        />
        <span>{field.unit}</span>
      </span>
    </label>
  );
}

export function Segmented({ name, label, required, options, value, onChange }: {
  name: string; label: string; required?: boolean; options: Option[]; value: string; onChange: (v: string) => void;
}) {
  const id = useId();
  return (
    <div className="field">
      <span className="field-label" id={id}>
        {label} {required && <span className="req">required</span>}
      </span>
      <div className="segmented" role="radiogroup" aria-labelledby={id}>
        {options.map((o) => (
          <label key={o.value || "unknown"}>
            <input type="radio" name={name} value={o.value} checked={value === o.value}
                   onChange={() => onChange(o.value)} />
            <span>{o.label}</span>
          </label>
        ))}
      </div>
    </div>
  );
}

export function ChoiceList({ name, label, options, value, onChange }: {
  name: string; label: string; options: Option[]; value: string; onChange: (v: string) => void;
}) {
  return (
    <div className="choice-list" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <label key={o.value}>
          <input type="radio" name={name} value={o.value} checked={value === o.value} onChange={() => onChange(o.value)} />
          <span><strong>{o.label}</strong>{o.detail}</span>
        </label>
      ))}
    </div>
  );
}

export function Select({ name, label, options, value, onChange }: {
  name: string; label: string; options: Option[]; value: string; onChange: (v: string) => void;
}) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      <select name={name} value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => <option key={o.value || "unknown"} value={o.value}>{o.label}</option>)}
      </select>
    </label>
  );
}
