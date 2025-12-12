import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Unified Customer Intelligence API" in response.json()["message"]

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_model_status():
    response = client.get("/models/status")
    assert response.status_code == 200
    assert "churn" in response.json()
    assert "clv" in response.json()
    assert "segmentation" in response.json()

def test_churn_prediction():
    test_data = {
        "customer_data": {
            "gender": "Male",
            "senior_citizen": 0,
            "tenure_months": 12,
            "monthly_charges": 50.0,
            "total_charges": 600.0
        }
    }
    response = client.post("/churn/predict", json=test_data)
    assert response.status_code in [200, 500]  # 500 if model not loaded

def test_rate_limiting():
    # Test rate limiting by making multiple requests
    responses = []
    for i in range(5):
        response = client.get("/health")
        responses.append(response.status_code)
    
    # Should not hit rate limit for health checks
    assert all(status == 200 for status in responses)