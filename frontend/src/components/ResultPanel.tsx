import { useEffect, useRef, useState } from "react";
import type { Assessment, Driver, Severity } from "../types";

const TIER_TEXT = { low: "Low risk", intermediate: "Intermediate risk", high: "High risk" } as const;
const pct = (p: number) => `${Math.round(p * 100)}%`;

export function ResultPanel({ result }: { result: Assessment | null }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (result && window.matchMedia("(max-width: 960px)").matches) ref.current?.scrollIntoView({ behavior: "smooth" });
  }, [result]);

  if (!result) {
    return (
      <aside className="result" aria-live="polite">
        <div className="paper">
          <h2>Result</h2>
          <p className="empty-copy">Fill in the form and choose <strong>Estimate risk</strong>. You'll see the estimated
            chance of coronary disease, what to do next, and which inputs moved the estimate.</p>
          <svg className="flatline" viewBox="0 0 400 60" aria-hidden="true"><path d="M0 30h400" /></svg>
        </div>
      </aside>
    );
  }

  const rec = result.recommendation;
  const caveats = [...result.warnings];
  if (result.missing_fields.length && result.missing_fields.length < 5) {
    caveats.push(`Not provided (averaged over): ${result.missing_fields.join(", ")}.`);
  }

  return (
    <aside className="result" aria-live="polite">
      <div className="paper" ref={ref}>
        <p className={`tier-name ${result.tier}`}>{TIER_TEXT[result.tier]}</p>
        <p className="risk">{result.risk_percent.toFixed(0)}<span className="risk-unit">%</span></p>
        <p className="risk-caption">estimated chance of significant coronary narrowing</p>

        <RiskStrip probability={result.probability} low={result.thresholds.low} high={result.thresholds.high} />

        <div className="next">
          <h3>What to do next</h3>
          <dl className="referral">
            <div><dt>See</dt><dd>{rec.referral}</dd></div>
            <div><dt>When</dt><dd>{rec.urgency}</dd></div>
          </dl>
          <ul>{rec.next_steps.map((s) => <li key={s}>{s}</li>)}</ul>
          {rec.risk_factor_notes.length > 0 && (
            <>
              <h4>Worth noting from the inputs</h4>
              <ul>{rec.risk_factor_notes.map((s) => <li key={s}>{s}</li>)}</ul>
            </>
          )}
        </div>

        <div className="drivers">
          <h3>What moved the estimate</h3>
          <p className="hint">Bars to the right raised the risk, bars to the left lowered it (SHAP values).</p>
          <DriverList drivers={result.drivers} />
        </div>

        {result.severity_if_present && result.tier !== "low" && <SeverityBars severity={result.severity_if_present} />}

        {caveats.length > 0 && (
          <div className="caveats">
            <h4>About this estimate</h4>
            <ul>{caveats.map((s) => <li key={s}>{s}</li>)}</ul>
          </div>
        )}

        <p className="red-flags">{rec.red_flags}</p>
        <p className="disclaimer">{rec.disclaimer}</p>
      </div>
    </aside>
  );
}

/** ECG-paper strip: zones are the model's real triage thresholds; the red trace marks this patient. */
function RiskStrip({ probability, low, high }: { probability: number; low: number; high: number }) {
  const [pos, setPos] = useState(0);
  useEffect(() => {
    setPos(0);
    const id = requestAnimationFrame(() => requestAnimationFrame(() => setPos(probability)));
    return () => cancelAnimationFrame(id);
  }, [probability]);

  return (
    <>
      <div className="strip" role="img"
           aria-label={`Risk ${Math.round(probability * 100)}% on a scale where below ${pct(low)} is low and from ${pct(high)} is high`}>
        <div className="zone low" style={{ flex: `0 0 ${low * 100}%` }}><span>Low</span></div>
        <div className="zone mid" style={{ flex: `0 0 ${(high - low) * 100}%` }}><span>Intermediate</span></div>
        <div className="zone high" style={{ flex: "1 1 auto" }}><span>High</span></div>
        <div className="marker" style={{ left: `${pos * 100}%` }} />
      </div>
      <p className="strip-scale">
        <span>0%</span>
        <span style={{ left: `${low * 100}%` }}>{pct(low)}</span>
        <span style={{ left: `${high * 100}%` }}>{pct(high)}</span>
        <span>100%</span>
      </p>
    </>
  );
}

function DriverList({ drivers }: { drivers: Driver[] }) {
  const maxAbs = Math.max(...drivers.map((d) => Math.abs(d.contribution)), 0.5);
  return (
    <ol className="driver-list">
      {drivers.map((d) => (
        <li key={d.feature}>
          <span className="driver-name">
            {d.label}
            <span className="driver-value">{d.value}, {d.direction}</span>
          </span>
          <span className="driver-bar" aria-hidden="true">
            <i className={d.contribution > 0 ? "up" : "down"} style={{ width: `${(Math.abs(d.contribution) / maxAbs) * 50}%` }} />
          </span>
        </li>
      ))}
    </ol>
  );
}

function SeverityBars({ severity }: { severity: Severity }) {
  return (
    <div className="severity">
      <h3>If disease is present: likely grade</h3>
      <div className="severity-bars">
        {Object.entries(severity.probabilities).map(([name, p]) => (
          <div key={name} className={`sev-row${name === severity.most_likely ? " top" : ""}`}>
            <span>{name.replace("grade", "Grade")}</span>
            <span className="sev-track"><i style={{ width: `${p * 100}%` }} /></span>
            <span>{pct(p)}</span>
          </div>
        ))}
      </div>
      <p className="hint">{severity.note}</p>
    </div>
  );
}
