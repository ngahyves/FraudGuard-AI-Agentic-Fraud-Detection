from fastapi.testclient import TestClient
from api.main import app
import pytest

client = TestClient(app)

def test_read_health():
    """Verify health end point"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_endpoint():
    """Verify prediction endpoint."""
    payload = {
        "transaction_id": 1,
        "transaction": {
            "step": 1, "type": "PAYMENT", "amount": 50.0, 
            "category": "Retail", "hour": 14, "day_of_week": 2,
            "day_of_month": 1, "month": 1
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "top_factors" in data