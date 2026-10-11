"""Authentication, browser sessions and rate limiting.

Two ways to call the prediction endpoints - and every prediction endpoint uses the same rule:
    1. Programmatic clients: `X-API-Key: <CARDIOFED_API_KEY>`
    2. The bundled web UI: a short-lived signed session cookie from POST /api/session,
       issued only to same-origin browser requests.

Fixes from the earlier project:
    - one auth dependency for all prediction routes (no unprotected twin endpoint)
    - the server starts in development without a key (a random key is generated and logged),
      and refuses to start in production without one - instead of crashing on a fresh clone
    - session tokens are HMAC-signed with an expiry, so they survive restarts and work with
      several worker processes (a per-process random token did not)
    - sessions are not limited to 127.0.0.1, so the UI also works inside Docker
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

log = logging.getLogger("cardiofed")
SESSION_COOKIE = "cardiofed_session"


@dataclass
class Settings:
    env: str
    api_key: str
    session_secret: bytes
    allowed_origins: list[str]
    ui_sessions: bool
    session_ttl: int
    rate_limit_per_minute: int

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.getenv("CARDIOFED_ENV", "development").lower()
        key = (os.getenv("CARDIOFED_API_KEY") or "").strip()
        if not key:
            if env == "production":
                raise RuntimeError("CARDIOFED_API_KEY must be set when CARDIOFED_ENV=production "
                                   "(generate one: python -c \"import secrets; print(secrets.token_urlsafe(32))\")")
            key = secrets.token_urlsafe(32)
            log.warning("CARDIOFED_API_KEY not set - generated a temporary development key: %s", key)
        elif len(key) < 16:
            raise RuntimeError("CARDIOFED_API_KEY must be at least 16 characters")
        secret = os.getenv("CARDIOFED_SESSION_SECRET") or key
        origins = [o.strip() for o in os.getenv("CARDIOFED_ALLOWED_ORIGINS", "").split(",") if o.strip()]
        return cls(env=env, api_key=key, session_secret=hashlib.sha256(secret.encode()).digest(),
                   allowed_origins=origins,
                   ui_sessions=os.getenv("CARDIOFED_UI_SESSIONS", "1") != "0",
                   session_ttl=int(os.getenv("CARDIOFED_SESSION_TTL", "3600")),
                   rate_limit_per_minute=int(os.getenv("CARDIOFED_RATE_LIMIT", "60")))


# ------------------------------------------------------------------ signed sessions
def make_session_token(settings: Settings, now: float | None = None) -> str:
    expires = int((now or time.time()) + settings.session_ttl)
    nonce = secrets.token_urlsafe(8)
    payload = f"{expires}.{nonce}"
    sig = hmac.new(settings.session_secret, payload.encode(), hashlib.sha256).digest()
    return f"{payload}.{base64.urlsafe_b64encode(sig).decode().rstrip('=')}"


def verify_session_token(settings: Settings, token: str | None, now: float | None = None) -> bool:
    if not token or token.count(".") != 2:
        return False
    expires, nonce, sig = token.split(".")
    payload = f"{expires}.{nonce}"
    good = base64.urlsafe_b64encode(hmac.new(settings.session_secret, payload.encode(),
                                             hashlib.sha256).digest()).decode().rstrip("=")
    if not hmac.compare_digest(sig, good):
        return False
    return expires.isdigit() and int(expires) > (now or time.time())


def is_same_origin(request: Request) -> bool:
    """Browser same-origin check: Sec-Fetch-Site if sent, otherwise Origin vs Host."""
    site = request.headers.get("sec-fetch-site")
    if site is not None:
        return site == "same-origin"
    origin = request.headers.get("origin")
    if origin is None:
        return False
    host = request.headers.get("host", "")
    return origin.split("://", 1)[-1] == host


# ------------------------------------------------------------------ rate limiting
class RateLimiter:
    """Sliding-window limiter per client (API key or session or IP). In-memory: fine for one
    process; use Redis or the API gateway's limiter when running several replicas."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, client: str) -> None:
        if self.per_minute <= 0:
            return
        now = time.monotonic()
        with self.lock:
            q = self.hits[client]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                raise HTTPException(429, "Rate limit exceeded - try again in a minute",
                                    headers={"Retry-After": "60"})
            q.append(now)


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_auth(request: Request, key: str | None = Security(api_key_header)) -> str:
    """Single dependency used by every prediction route. Returns the client identity."""
    settings: Settings = request.app.state.settings
    limiter: RateLimiter = request.app.state.limiter
    if key and hmac.compare_digest(key, settings.api_key):
        ident = "key:" + hashlib.sha256(key.encode()).hexdigest()[:12]
    elif verify_session_token(settings, request.cookies.get(SESSION_COOKIE)):
        ident = "ui:" + (request.client.host if request.client else "unknown")
    else:
        raise HTTPException(401, "Missing or invalid API key (send X-API-Key) or UI session")
    limiter.check(ident)
    return ident
