"""CardioFed API + web UI.

    uvicorn cardiofed.api.main:app --reload            # http://localhost:8000  (UI)
                                                       # http://localhost:8000/docs (OpenAPI)
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from cardiofed import __version__
from cardiofed import config as C
from cardiofed.api.schemas import AssessmentOut, BatchIn, BatchOut, PatientIn
from cardiofed.api.security import (SESSION_COOKIE, RateLimiter, Settings, is_same_origin,
                                    make_session_token, require_auth)
from cardiofed.predictor import Predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("cardiofed")
FRONTEND = C.PROJECT_ROOT / "frontend" / "dist"   # React build output (npm run build)

CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
       "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'")


def model_summary(predictor: Predictor) -> dict:
    meta = predictor.meta
    fed = None
    fed_path = C.REPORTS / "federated.json"
    if fed_path.is_file():
        rep = json.loads(fed_path.read_text())
        fed = {name: {"hospital_mean_auc": r["hospital_mean"],
                      **{s: r.get(s) for s in C.HOSPITALS}} for name, r in rep["results"].items()}
    return {
        "version": meta.get("version"), "trained_at": meta.get("trained_at"), "data": meta.get("data"),
        "thresholds": predictor.model.thresholds, "params": meta.get("params"),
        "software": meta.get("software"), "metrics": meta.get("metrics"), "federated": fed,
        "hospitals": {k: v[0] for k, v in C.HOSPITALS.items()},
        "features": {f: C.FEATURE_LABELS[f] for f in meta.get("features", [])},
        "categories": C.CATEGORY_CODES, "valid_ranges": C.VALID_RANGES,
    }


def create_app(settings: Settings | None = None, predictor: Predictor | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.limiter = RateLimiter(settings.rate_limit_per_minute)
        app.state.predictor = predictor or Predictor()
        app.state.summary = model_summary(app.state.predictor)
        log.info("model %s loaded (env=%s)", app.state.predictor.meta.get("version"), app.state.settings.env)
        yield

    app = FastAPI(title="CardioFed API", version=__version__, lifespan=lifespan,
                  description="Heart disease risk estimation trained on the UCI 4-hospital dataset. "
                              "Prediction routes need `X-API-Key` (or the web UI's session cookie). "
                              "Educational demo - not a medical device.")

    if settings.allowed_origins:  # cross-origin access is opt-in; the bundled UI is same-origin
        app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins, allow_methods=["GET", "POST"],
                           allow_headers=["Content-Type", "X-API-Key"])

    @app.middleware("http")
    async def headers_and_logging(request: Request, call_next):
        rid = uuid.uuid4().hex[:8]
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if not request.url.path.startswith(("/docs", "/redoc", "/openapi")):
            response.headers["Content-Security-Policy"] = CSP
        if request.url.path.startswith("/api/"):  # never log request bodies (patient data)
            log.info("%s %s %s %.0fms rid=%s", request.method, request.url.path, response.status_code,
                     (time.perf_counter() - t0) * 1000, rid)
        return response

    @app.get("/api/health", tags=["meta"])
    def health(request: Request):
        """Liveness/readiness - no auth, for load balancers."""
        return {"status": "ok", "model_version": request.app.state.predictor.meta.get("version")}

    @app.get("/api/model", tags=["meta"])
    def model_info(request: Request):
        """Model card: data, metrics (held-out test, external validation, federated), thresholds."""
        return request.app.state.summary

    @app.post("/api/session", status_code=204, tags=["auth"])
    def create_session(request: Request, response: Response):
        """Issue a signed, short-lived session cookie to the bundled web UI (same-origin only)."""
        s: Settings = request.app.state.settings
        if not s.ui_sessions:
            raise HTTPException(403, "Browser sessions are disabled on this server")
        if not is_same_origin(request):
            raise HTTPException(403, "Sessions are only issued to the same-origin web UI")
        response.set_cookie(SESSION_COOKIE, make_session_token(s), max_age=s.session_ttl, httponly=True,
                            samesite="strict", secure=request.url.scheme == "https", path="/api")

    @app.post("/api/predict", response_model=AssessmentOut, tags=["predict"])
    def predict(patient: PatientIn, request: Request, _client: str = Depends(require_auth)):
        """Risk estimate, triage tier, recommendation and explanation for one patient."""
        return request.app.state.predictor.assess(patient.model_dump())

    @app.post("/api/predict/batch", response_model=BatchOut, tags=["predict"])
    def predict_batch(batch: BatchIn, request: Request, _client: str = Depends(require_auth)):
        """Up to 100 patients in one call."""
        return {"results": request.app.state.predictor.assess_many([p.model_dump() for p in batch.patients])}

    if C.FIGURES.is_dir():  # charts used by the "How the model was built" page
        app.mount("/figures", StaticFiles(directory=C.FIGURES), name="figures")
    if (FRONTEND / "index.html").is_file():
        app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
    else:
        log.warning("frontend/dist not found - run `npm install && npm run build` in frontend/ to serve the web UI")
    return app


app = create_app()
