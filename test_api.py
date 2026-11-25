from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_model_is_not_falsely_reported_as_available():
    response = client.get("/v1/model/status")
    assert response.status_code == 200
    assert response.json()["available"] is False
