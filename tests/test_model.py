import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import mlflow


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEST_MODEL_PATH = PROJECT_ROOT / "configs" / "best_model.yaml"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "amazon_sales_regression.csv"


class TestModelLoading:
    """Tests for model loading."""
    
    def test_best_model_config_exists(self):
        """Best model config file should exist."""
        assert BEST_MODEL_PATH.exists(), "best_model.yaml not found. Run select_best_regression_run.py first."
    
    def test_can_load_model_from_mlflow(self):
        """Should be able to load model from MLflow."""
        import yaml
        
        with open(BEST_MODEL_PATH) as f:
            config = yaml.safe_load(f)
        
        model_uri = config["regression"]["model_uri"]
        model = mlflow.sklearn.load_model(model_uri)
        
        assert model is not None
    
    def test_model_has_predict_method(self):
        """Loaded model should have predict method."""
        import yaml
        
        with open(BEST_MODEL_PATH) as f:
            config = yaml.safe_load(f)
        
        model_uri = config["regression"]["model_uri"]
        model = mlflow.sklearn.load_model(model_uri)
        
        assert hasattr(model, "predict")


class TestModelPrediction:
    """Tests for model predictions."""
    
    @pytest.fixture
    def model(self):
        """Load the model for testing."""
        import yaml
        
        with open(BEST_MODEL_PATH) as f:
            config = yaml.safe_load(f)
        
        model_uri = config["regression"]["model_uri"]
        return mlflow.sklearn.load_model(model_uri)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample input data."""
        return pd.DataFrame([{
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
        }])
    
    def test_model_returns_prediction(self, model, sample_data):
        """Model should return a prediction."""
        prediction = model.predict(sample_data)
        
        assert prediction is not None
        assert len(prediction) == 1
    
    def test_prediction_is_numeric(self, model, sample_data):
        """Prediction should be a number."""
        prediction = model.predict(sample_data)
        
        # Accept both Python and numpy numeric types
        assert np.issubdtype(type(prediction[0]), np.number)
    
    def test_prediction_is_positive(self, model, sample_data):
        """Prediction (TotalAmount) should be positive."""
        prediction = model.predict(sample_data)
        
        assert prediction[0] > 0
    
    def test_batch_prediction(self, model, sample_data):
        """Model should handle batch predictions."""
        batch_data = pd.concat([sample_data] * 5, ignore_index=True)
        
        predictions = model.predict(batch_data)
        
        assert len(predictions) == 5


class TestDataFiles:
    """Tests for data files."""
    
    def test_processed_data_exists(self):
        """Processed data file should exist."""
        assert DATA_PATH.exists(), "Processed data not found."
    
    def test_processed_data_has_required_columns(self):
        """Processed data should have required columns."""
        df = pd.read_csv(DATA_PATH)
        
        required_columns = [
            "Category", "Brand", "Quantity", "UnitPrice", 
            "Discount", "Tax", "ShippingCost", "TotalAmount",
            "PaymentMethod", "OrderStatus", "City", "State", "Country",
            "OrderYear", "OrderMonth", "OrderDayOfWeek"
        ]
        
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_processed_data_not_empty(self):
        
        df = pd.read_csv(DATA_PATH)
        
        assert len(df) > 0

