import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import requests


@dataclass
class ModelInfo:
    source: str  # "registry" or "best_model_yaml"
    model_uri: str
    alias: Optional[str] = None
    model_name: Optional[str] = None
    version: Optional[str] = None
    run_id: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class AvailableModel:
    alias: str
    version: str
    run_id: str
    metrics: Dict[str, float]
    description: str = ""


@dataclass
class AvailableModels:
    registered_models: List[AvailableModel]
    fallback_available: bool


@dataclass
class PredictionResult:
    success: bool
    prediction: Optional[float] = None
    predictions: Optional[List[float]] = None
    error: Optional[str] = None


class PredictorClient:
    """Client for the prediction API with model selection support."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
    
    def health_check(self) -> bool:
        """Check if the API is running."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def get_available_models(self) -> Optional[AvailableModels]:
        """Get list of available models (champion, challenger, fallback)."""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [
                    AvailableModel(
                        alias=m["alias"],
                        version=m["version"],
                        run_id=m["run_id"],
                        metrics=m.get("metrics", {}),
                        description=m.get("description", ""),
                    )
                    for m in data.get("registered_models", [])
                ]
                return AvailableModels(
                    registered_models=models,
                    fallback_available=data.get("fallback_available", False),
                )
            return None
        except requests.RequestException:
            return None
    
    def load_model(self, alias: str) -> tuple[bool, str]:
        """Load a model by alias (champion, challenger, or fallback)."""
        try:
            response = requests.post(
                f"{self.base_url}/models/load",
                params={"alias": alias},
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()
                return True, data.get("message", "Model loaded")
            else:
                detail = response.json().get("detail", "Unknown error")
                return False, detail
        except requests.RequestException as e:
            return False, str(e)
    
    def get_model_info(self) -> Optional[ModelInfo]:
        """Get information about the currently loaded model."""
        try:
            response = requests.get(f"{self.base_url}/model-info", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return ModelInfo(
                    source=data.get("source", "unknown"),
                    model_uri=data.get("model_uri", ""),
                    alias=data.get("alias"),
                    model_name=data.get("model_name"),
                    version=data.get("version"),
                    run_id=data.get("run_id"),
                    metrics=data.get("metrics", {}),
                )
            return None
        except requests.RequestException:
            return None
    
    def predict(self, features: Dict[str, Any]) -> PredictionResult:
        """Make a single prediction."""
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json=features,
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()
                return PredictionResult(
                    success=True,
                    prediction=data["predicted_total_amount"],
                )
            else:
                return PredictionResult(
                    success=False,
                    error=response.json().get("detail", "Unknown error"),
                )
        except requests.RequestException as e:
            return PredictionResult(success=False, error=str(e))
    
    def predict_batch(self, features_list: List[Dict[str, Any]]) -> PredictionResult:
        """Make batch predictions."""
        try:
            response = requests.post(
                f"{self.base_url}/predict_batch",
                json=features_list,
                timeout=60,
            )
            if response.status_code == 200:
                data = response.json()
                return PredictionResult(
                    success=True,
                    predictions=data["predictions"],
                )
            else:
                return PredictionResult(
                    success=False,
                    error=response.json().get("detail", "Unknown error"),
                )
        except requests.RequestException as e:
            return PredictionResult(success=False, error=str(e))


def get_client(base_url: str | None = None) -> PredictorClient:
    # Get a predictor client instance (uses API_URL environment variable if set, otherwise defaults to localhost)
    if base_url is None:
        base_url = os.environ.get("API_URL", "http://localhost:8000")
    return PredictorClient(base_url)
