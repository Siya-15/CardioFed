import os

import pytest

os.environ.setdefault("CARDIOFED_API_KEY", "test-api-key-0123456789")

from cardiofed.predictor import Predictor  # noqa: E402

API_KEY = os.environ["CARDIOFED_API_KEY"]

FULL_PATIENT = {"age": 61, "sex": "male", "cp": "asymptomatic", "trestbps": 148, "chol": 262, "fbs": "yes",
                "restecg": "lv_hypertrophy", "thalach": 112, "exang": "yes", "oldpeak": 2.4, "slope": "flat"}
LOW_PATIENT = {"age": 44, "sex": "female", "cp": "non_anginal_pain", "trestbps": 118, "chol": 205, "fbs": "no",
               "restecg": "normal", "thalach": 170, "exang": "no", "oldpeak": 0.0, "slope": "upsloping"}


@pytest.fixture(scope="session")
def predictor():
    return Predictor()


@pytest.fixture()
def make_client(predictor):
    """Factory: TestClient for an app built with custom settings."""
    from fastapi.testclient import TestClient

    from cardiofed.api.main import create_app
    from cardiofed.api.security import Settings

    clients = []

    def _make(**overrides):
        s = Settings.from_env()
        for k, v in overrides.items():
            setattr(s, k, v)
        c = TestClient(create_app(settings=s, predictor=predictor))
        c.__enter__()
        clients.append(c)
        return c

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture()
def client(make_client):
    return make_client()
