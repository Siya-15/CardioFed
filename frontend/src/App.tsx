import { useEffect, useState } from "react";
import { ApiError, getVersion, predict, warmUpSession } from "./api";
import { ModelView } from "./components/ModelView";
import { PatientForm } from "./components/PatientForm";
import { ResultPanel } from "./components/ResultPanel";
import type { Assessment, PatientInput } from "./types";

type View = "assess" | "model";
const viewFromHash = (): View => (window.location.hash === "#model" ? "model" : "assess");

export function App() {
  const [view, setView] = useState<View>(viewFromHash);
  const [result, setResult] = useState<Assessment | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [version, setVersion] = useState<string | null>(null);

  useEffect(() => {
    const onHash = () => setView(viewFromHash());
    window.addEventListener("hashchange", onHash);
    warmUpSession();
    getVersion().then(setVersion).catch(() => {});
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  async function assess(patient: PatientInput) {
    setBusy(true);
    setError(null);
    try {
      setResult(await predict(patient));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not reach the server. Check that it is running.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <header className="topbar">
        <a className="brand" href="#assess" aria-label="CardioFed home">
          <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden="true"><path d="M2 17h7l3-8 5 15 4-11 2 4h7" /></svg>
          CardioFed
        </a>
        <nav className="tabs" aria-label="Sections">
          <a href="#assess" aria-current={view === "assess" ? "page" : undefined}>Assess a patient</a>
          <a href="#model" aria-current={view === "model" ? "page" : undefined}>How it was built</a>
        </nav>
      </header>

      <main>
        {view === "assess" ? (
          <section className="view assess" aria-label="Assess a patient">
            <PatientForm onSubmit={assess} busy={busy} serverError={error} />
            <ResultPanel result={result} />
          </section>
        ) : (
          <ModelView />
        )}
      </main>

      <footer className="footer">
        <p>CardioFed{version ? ` v${version}` : ""}. Data: Janosi, Steinbrunn, Pfisterer &amp; Detrano, UCI Machine Learning
          Repository (CC BY 4.0). API documentation at <a href="/docs">/docs</a>.</p>
      </footer>
    </>
  );
}
