import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.main import app  # noqa: E402

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "models_loaded": True}


def test_predict_valid_ticket(client):
    r = client.post("/predict", json={"text": "I was charged twice for my subscription"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"category", "priority", "confidence"}
    assert body["category"] == "Billing"
    assert body["priority"] in {"High", "Medium", "Low"}
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_other_categories(client):
    cases = {
        "the app keeps crashing when I open it": "Technical",
        "I forgot my password and cannot log in": "Account",
        "my order has not arrived yet": "Shipping",
    }
    for text, expected in cases.items():
        r = client.post("/predict", json={"text": text})
        assert r.status_code == 200
        assert r.json()["category"] == expected


def test_predict_empty_text_returns_422(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_predict_whitespace_only_returns_422(client):
    assert client.post("/predict", json={"text": "     "}).status_code == 422


def test_predict_missing_text_field_returns_422(client):
    assert client.post("/predict", json={}).status_code == 422


def test_predict_wrong_type_returns_422(client):
    assert client.post("/predict", json={"text": 123}).status_code == 422


def test_predict_batch(client):
    payload = {"tickets": [
        {"text": "I was charged twice for my subscription"},
        {"text": "my order has not arrived yet"},
    ]}
    r = client.post("/predict-batch", json=payload)
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2
    assert preds[0]["category"] == "Billing"
    assert preds[1]["category"] == "Shipping"

def test_predict_batch_empty_list_returns_422(client):
    assert client.post("/predict-batch", json={"tickets": []}).status_code == 422
