import pytest

from cardiofed.api.security import Settings, make_session_token, verify_session_token
from tests.conftest import API_KEY, FULL_PATIENT

AUTH = {"X-API-Key": API_KEY}
SAME_ORIGIN = {"Sec-Fetch-Site": "same-origin"}


def test_health_and_model_card_are_public(client):
    assert client.get("/api/health").json()["status"] == "ok"
    card = client.get("/api/model").json()
    assert card["data"]["patients"] == 918
    assert set(card["metrics"]["leave_one_hospital_out"]) == {"cleveland", "hungarian", "switzerland", "va"}


def test_prediction_requires_auth(client):
    assert client.post("/api/predict", json=FULL_PATIENT).status_code == 401
    assert client.post("/api/predict", json=FULL_PATIENT, headers={"X-API-Key": "wrong-key-000000000"}).status_code == 401
    assert client.post("/api/predict/batch", json={"patients": [FULL_PATIENT]}).status_code == 401


def test_prediction_with_api_key(client):
    r = client.post("/api/predict", json=FULL_PATIENT, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["tier"] == "high" and body["recommendation"]["referral"].startswith("Cardiologist")
    assert r.headers["Content-Security-Policy"].startswith("default-src 'self'")


def test_minimal_patient_lists_missing_fields(client):
    r = client.post("/api/predict", json={"age": 50, "sex": "female", "cp": "typical_angina"}, headers=AUTH)
    assert r.status_code == 200
    assert set(r.json()["missing_fields"]) == {"trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope"}


@pytest.mark.parametrize("patch", [
    {"cholesterol": 200},               # unknown field (typo) is rejected, not silently ignored
    {"chol": 0},                         # impossible value
    {"sex": "M"},                        # values must use the documented vocabulary
    {"age": None},                       # required
    {"oldpeak": 12},
])
def test_invalid_input_is_422(client, patch):
    assert client.post("/api/predict", json={**FULL_PATIENT, **patch}, headers=AUTH).status_code == 422


def test_batch(client):
    r = client.post("/api/predict/batch", json={"patients": [FULL_PATIENT] * 3}, headers=AUTH)
    assert r.status_code == 200 and len(r.json()["results"]) == 3
    assert client.post("/api/predict/batch", json={"patients": [FULL_PATIENT] * 101}, headers=AUTH).status_code == 422


def test_ui_session_flow(client):
    assert client.post("/api/session").status_code == 403                                  # no browser headers
    assert client.post("/api/session", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert client.post("/api/session", headers=SAME_ORIGIN).status_code == 204
    assert client.post("/api/predict", json=FULL_PATIENT).status_code == 200                # cookie now works
    client.cookies.set("cardiofed_session", "123.abc.forged", path="/api")
    assert client.post("/api/predict", json=FULL_PATIENT).status_code == 401


def test_sessions_can_be_disabled(make_client):
    c = make_client(ui_sessions=False)
    assert c.post("/api/session", headers=SAME_ORIGIN).status_code == 403


def test_session_tokens_expire():
    s = Settings.from_env()
    token = make_session_token(s, now=1_000)
    assert verify_session_token(s, token, now=1_000 + s.session_ttl - 1)
    assert not verify_session_token(s, token, now=1_000 + s.session_ttl + 1)


def test_rate_limit(make_client):
    c = make_client(rate_limit_per_minute=3)
    codes = [c.post("/api/predict", json=FULL_PATIENT, headers=AUTH).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]


def test_production_requires_api_key(monkeypatch):
    monkeypatch.setenv("CARDIOFED_ENV", "production")
    monkeypatch.delenv("CARDIOFED_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="must be set"):
        Settings.from_env()
    monkeypatch.setenv("CARDIOFED_API_KEY", "short")
    with pytest.raises(RuntimeError, match="at least 16"):
        Settings.from_env()


def test_frontend_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "CardioFed" in r.text
    assert client.get("/figures/roc_test.png").status_code == 200
