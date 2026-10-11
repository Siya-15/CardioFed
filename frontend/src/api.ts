import type { Assessment, ModelSummary, PatientInput } from "./types";

// The bundled UI authenticates with a short-lived, signed session cookie from POST /api/session
// (same-origin only). Programmatic clients use the X-API-Key header instead.
let session: Promise<void> | null = null;

function ensureSession(): Promise<void> {
  if (!session) {
    session = fetch("/api/session", { method: "POST", credentials: "same-origin" })
      .then((r) => {
        if (!r.ok) throw new Error("This server does not allow browser sessions. Use the API with an API key.");
      })
      .catch((e) => {
        session = null;
        throw e;
      });
  }
  return session;
}

export class ApiError extends Error {}

function describeError(status: number, data: { detail?: unknown }): string {
  if (status === 422 && Array.isArray(data.detail)) {
    return data.detail
      .map((d: { loc?: string[]; msg?: string }) => `${(d.loc ?? []).slice(-1)[0]}: ${d.msg}`)
      .join(". ");
  }
  if (status === 429) return "Too many requests. Wait a minute and try again.";
  return typeof data.detail === "string" ? data.detail : `The server returned an error (${status}).`;
}

export async function predict(patient: PatientInput): Promise<Assessment> {
  await ensureSession();
  const opts: RequestInit = {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patient),
  };
  let res = await fetch("/api/predict", opts);
  if (res.status === 401) {
    // session expired: get a fresh one and retry once
    session = null;
    await ensureSession();
    res = await fetch("/api/predict", opts);
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(describeError(res.status, data));
  return data as Assessment;
}

export async function getModel(): Promise<ModelSummary> {
  const res = await fetch("/api/model");
  if (!res.ok) throw new ApiError("Model details could not be loaded.");
  return res.json();
}

export async function getVersion(): Promise<string | null> {
  const res = await fetch("/api/health");
  return res.ok ? ((await res.json()).model_version as string) : null;
}

export function warmUpSession(): void {
  ensureSession().catch(() => {});
}
