import pytest
from fastapi.testclient import TestClient

from amazon_sales_ml.api import app as app_module


# Sample input data for testing
SAMPLE_INPUT = {
    "Category": "Electronics",
    "Brand": "Zenith",
    "Quantity": 2,
    "UnitPrice": 299.99,
    "Discount": 0.1,
    "Tax": 48.0,
    "ShippingCost": 5.99,
    "PaymentMethod": "Credit Card",
    "OrderStatus": "Delivered",
    "City": "New York",
    "State": "NY",
    "Country": "United States",
    "OrderYear": 2024,
    "OrderMonth": 6,
    "OrderDayOfWeek": "Monday"
}


@pytest.fixture(scope="module")
def client():
    """Create test client with model loaded."""
    # Manually trigger model loading
    app_module.startup_event()
    
    # Create client
    with TestClient(app_module.app) as c:
        yield c


class TestHealthEndpoint:
    """Tests for the /health endpoint."""
    
    def test_health_returns_ok(self, client):
        """Health endpoint should return status ok."""
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestModelInfoEndpoint:
    """Tests for the /model-info endpoint."""
    
    def test_model_info_returns_data(self, client):
        """Model info endpoint should return model details."""
        response = client.get("/model-info")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields exist
        assert "source" in data
        assert "model_uri" in data


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""
    
    def test_predict_returns_prediction(self, client):
        """Predict endpoint should return a prediction."""
        response = client.post("/predict", json=SAMPLE_INPUT)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check prediction exists and is a number
        assert "predicted_total_amount" in data
        assert isinstance(data["predicted_total_amount"], (int, float))
    
    def test_predict_returns_positive_value(self, client):
        """Prediction should be a positive value."""
        response = client.post("/predict", json=SAMPLE_INPUT)
        
        data = response.json()
        assert data["predicted_total_amount"] > 0
    
    def test_predict_missing_field_returns_error(self, client):
        """Missing required field should return 422 error."""
        incomplete_input = SAMPLE_INPUT.copy()
        del incomplete_input["Category"]
        
        response = client.post("/predict", json=incomplete_input)
        
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Tests for the /predict_batch endpoint."""
    
    def test_batch_predict_returns_predictions(self, client):
        """Batch predict should return list of predictions."""
        batch_input = [SAMPLE_INPUT, SAMPLE_INPUT]
        
        response = client.post("/predict_batch", json=batch_input)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions" in data
        assert len(data["predictions"]) == 2
    
    def test_batch_predict_empty_list_returns_error(self, client):
        """Empty input list should return 400 error."""
        response = client.post("/predict_batch", json=[])
        
        assert response.status_code == 400


class TestModelsEndpoint:
    """Tests for the /models endpoint."""
    
    def test_models_returns_list(self, client):
        """Models endpoint should return available models."""
        response = client.get("/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "registered_models" in data
        assert "fallback_available" in data
        assert isinstance(data["registered_models"], list)
