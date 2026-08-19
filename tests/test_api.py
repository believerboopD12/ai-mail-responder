import os
os.environ.pop("OPENAI_API_KEY", None)
os.environ["DATABASE_URL"] = "sqlite:///./test_emails.db"

import pytest
from fastapi.testclient import TestClient
from app.classifier import IntentClassifier
from app.database import Base, engine
from app.main import app


@pytest.fixture(autouse=True)
def clean_database():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/health").json() == {"status": "healthy"}


def test_root_points_to_api_docs(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"

def test_valid_prediction(client):
    response = client.post("/predict", json={"subject": "Refund missing", "body": "My returned payment has not arrived."})
    assert response.status_code == 200
    assert response.json()["intent"] == "refund"
    assert 0 <= response.json()["confidence"] <= 1


@pytest.mark.parametrize("payload", [{"subject": "", "body": "Hello"}, {"subject": "Hi", "body": "   "}])
def test_empty_input_is_rejected(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_classifier_output():
    intent, confidence = IntentClassifier().predict("Cannot log in", "Password reset does not work")
    assert intent in {"inquiry", "support", "complaint", "refund", "sales", "spam"}
    assert isinstance(confidence, float)


def test_reply_uses_fallback_and_is_stored(client):
    response = client.post("/reply", json={"subject": "App error", "body": "The application keeps crashing."})
    assert response.status_code == 201
    assert "Customer Support" in response.json()["generated_reply"]
    assert client.get("/emails").json()["total"] == 1
    assert client.get(f"/emails/{response.json()['id']}").status_code == 200


def test_overdue_refund_is_high_priority(client):
    response = client.post(
        "/predict",
        json={"subject": "Refund overdue", "body": "My approved refund has not reached my bank."},
    )
    assert response.status_code == 200
    assert response.json()["intent"] == "refund"
    assert response.json()["priority"] == "high"

def test_spam_prediction_is_low_priority(client):
    response = client.post(
        "/predict",
        json={"subject": "Free prize", "body": "WINNER! Claim your cash reward now by calling this premium number."},
    )
    assert response.status_code == 200
    assert response.json()["intent"] == "spam"
    assert response.json()["priority"] == "low"
    assert 0 <= response.json()["confidence"] <= 1

def test_missing_email_returns_404(client):
    assert client.get("/emails/999").status_code == 404
