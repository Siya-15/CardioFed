import { useEffect, useState } from "react";
import { getModel } from "../api";
import type { ModelSummary } from "../types";

const fmt = (x: number | null | undefined, d = 3) => (x === null || x === undefined ? "-" : x.toFixed(d));
const pct = (p: number) => `${Math.round(p * 100)}%`;

export function ModelView() {
  const [m, setM] = useState<ModelSummary | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => { getModel().then(setM).catch(() => setError(true)); }, []);

  const lede = error ? "Model details could not be loaded."
    : !m ? "Loading model details…"
    : `XGBoost trained on ${m.data.patients} patients from four hospitals (${Math.round(m.data.prevalence * 100)}% with ` +
      "coronary disease). Every number below comes from patients the model did not train on.";
  const testN = m ? Object.values(m.metrics.test.per_hospital).reduce((a, h) => a + h.n, 0) : 0;

  return (
    <section className="view model" aria-labelledby="model-title">
      <div className="model-head">
        <h1 id="model-title">How the model was built</h1>
        <p>{lede}</p>
      </div>

      <div className="model-grid">
        <article>
          <h2>Held-out test set</h2>
          {m && <p className="hint">{testN} patients held out from training. Lower Brier is better.</p>}
          <table className="data">
            <thead><tr><th scope="col">Model</th><th scope="col">AUC</th><th scope="col">95% CI</th><th scope="col">Brier</th></tr></thead>
            <tbody>
              {m && Object.entries(m.metrics.test.models).map(([name, r]) => (
                <tr key={name} className={name.startsWith("xgboost") ? "ours" : undefined}>
                  <td>{name}</td><td>{fmt(r.auc ?? 0.5)}</td>
                  <td>{r.auc_ci95 ? `${fmt(r.auc_ci95[0])}-${fmt(r.auc_ci95[1])}` : "-"}</td><td>{fmt(r.brier)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="note">XGBoost and logistic regression are statistically tied on this small dataset; XGBoost was kept
            for its per-patient explanations and its clearly better severity grading.</p>
          <img src="/figures/roc_test.png" alt="ROC curves of XGBoost, logistic regression and random forest on the test set" loading="lazy" />
        </article>

        <article>
          <h2>Tested on a hospital it never saw</h2>
          <p className="hint">Train on three hospitals, test on the fourth.</p>
          <table className="data">
            <thead><tr><th scope="col">Held-out hospital</th><th scope="col">Patients</th><th scope="col">With disease</th>
              <th scope="col">XGBoost</th><th scope="col">Log. reg.</th></tr></thead>
            <tbody>
              {m && Object.entries(m.metrics.leave_one_hospital_out).map(([site, r]) => (
                <tr key={site}><td>{m.hospitals[site]}</td><td>{r.n}</td><td>{pct(r.prevalence)}</td>
                  <td>{fmt(r.xgboost_auc)}</td><td>{fmt(r.logreg_auc)}</td></tr>
              ))}
            </tbody>
          </table>
          <img src="/figures/loho_auc.png" alt="Bar chart of AUC on each unseen hospital" loading="lazy" />
        </article>

        <article>
          <h2>Federated learning across the four hospitals</h2>
          <p className="hint">Patient rows never leave a hospital; only statistics and model parameters are shared.
            Scores are the mean AUC over each hospital's own test patients.</p>
          <table className="data">
            <thead><tr><th scope="col">Strategy</th><th scope="col">Mean hospital AUC</th><th scope="col">VA Long Beach</th></tr></thead>
            <tbody>
              {m?.federated && Object.entries(m.federated).map(([name, r]) => (
                <tr key={name} className={name.startsWith("federated XGBoost") ? "ours" : undefined}>
                  <td>{name}</td><td>{fmt(r.hospital_mean_auc)}</td><td>{fmt(r.va)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <img src="/figures/federated.png" alt="Federated, local and centralised AUC, and FedAvg convergence" loading="lazy" />
        </article>

        <article>
          <h2>A shortcut we removed</h2>
          <p>Zurich recorded no cholesterol at all, and 94% of its patients had disease. A model that treats a blank field
            as information learns "no cholesterol means sick": leaving the box empty raised the estimate by about ten
            points. Training with randomly imputed values and averaging over them at prediction time removed the effect
            without losing accuracy on unseen hospitals.</p>
          <img src="/figures/missingness_shortcut.png" alt="Bar chart: risk change when cholesterol is left blank, before and after the fix" loading="lazy" />
        </article>

        <article>
          <h2>What drives the prediction overall</h2>
          <img src="/figures/shap_importance.png" alt="Mean absolute SHAP value per feature" loading="lazy" />
        </article>

        <article>
          <h2>Data and limits</h2>
          <ul className="plain">
            <li>UCI Heart Disease data: patients referred for coronary angiography at Cleveland, Budapest, Zurich/Basel and
              Long Beach VA in the 1980s. The outcome is more than 50% narrowing of a major vessel.</li>
            <li>Everyone in the data was already suspected of heart disease, so risk is higher than in the general
              population, and "no chest pain" behaves as a risk marker.</li>
            <li>Two tests (fluoroscopy vessel count and thallium scan) were dropped: they were recorded almost only at
              Cleveland, and fluoroscopy is part of the diagnosis itself.</li>
            <li>Not validated for clinical use.</li>
          </ul>
        </article>
      </div>
    </section>
  );
}
