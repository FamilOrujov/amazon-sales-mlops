import os
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import yaml
import mlflow
from mlflow import MlflowClient

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BEST_MODEL_PATH = PROJECT_ROOT / "configs" / "best_model.yaml"
REGISTRY_PATH = PROJECT_ROOT / "configs" / "registry.yaml"
EXPORTED_MODEL_PATH = PROJECT_ROOT / "models" / "champion.pkl"

# Docker mode detection
DOCKER_MODE = os.environ.get("DOCKER_MODE", "false").lower() == "true"


# Pydantic Models (Request/Response schemas)
class SaleInput(BaseModel):
    """Input features for a single sales prediction."""
    Category: str
    Brand: str
    Quantity: int
    UnitPrice: float
    Discount: float
    Tax: float
    ShippingCost: float
    PaymentMethod: str
    OrderStatus: str
    City: str
    State: str
    Country: str
    OrderYear: int
    OrderMonth: int
    OrderDayOfWeek: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response containing the predicted TotalAmount."""
    predicted_total_amount: float


class BatchPredictionResponse(BaseModel):
    """Response containing multiple predictions."""
    predictions: List[float]


class ModelInfoResponse(BaseModel):
    """Response containing model information."""
    source: str  # "registry" or "best_model_yaml"
    alias: Optional[str] = None
    model_name: Optional[str] = None
    version: Optional[str] = None
    run_id: Optional[str] = None
    model_uri: str
    metrics: Dict[str, float] = {}


class AvailableModelsResponse(BaseModel):
    """Response containing available models."""
    registered_models: List[Dict[str, Any]]
    fallback_available: bool


# Model Registry helpers
def load_registry_config() -> Dict[str, Any]:
    """Load registry configuration."""
    if not REGISTRY_PATH.exists():
        return {}
    with open(REGISTRY_PATH, "r") as f:
        return yaml.safe_load(f)


def get_registered_model_info(model_name: str, alias: str) -> Optional[Dict[str, Any]]:
    """Get model info from MLflow Model Registry by alias."""
    try:
        client = MlflowClient()
        
        # Get model version by alias
        mv = client.get_model_version_by_alias(model_name, alias)
        
        # Get run info for metrics
        run = client.get_run(mv.run_id)
        metrics = {k: v for k, v in run.data.metrics.items() if k.startswith("test_")}
        
        return {
            "model_name": model_name,
            "alias": alias,
            "version": mv.version,
            "run_id": mv.run_id,
            "model_uri": f"models:/{model_name}@{alias}",
            "metrics": metrics,
            "description": mv.description or "",
            "tags": mv.tags,
        }
    except Exception:
        return None


def load_model_by_alias(model_name: str, alias: str):
    """Load a model from the registry by alias."""
    model_uri = f"models:/{model_name}@{alias}"
    model = mlflow.sklearn.load_model(model_uri)
    info = get_registered_model_info(model_name, alias)
    return model, info


def load_exported_model():
    """Load model from exported pickle file (Docker mode)."""
    if not EXPORTED_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Exported model not found at {EXPORTED_MODEL_PATH}. "
            "Run `python scripts/export_model.py` first."
        )
    
    with open(EXPORTED_MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    
    model = model_data["model"]
    info = {
        "source": "exported_pickle",
        "model_uri": str(EXPORTED_MODEL_PATH),
        "run_id": model_data["info"].get("run_id"),
        "metrics": {model_data["info"]["metric"]: model_data["info"]["metric_value"]},
    }
    
    return model, info


def load_fallback_model():
    """Load model from best_model.yaml (fallback)."""
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Best model config not found at {BEST_MODEL_PATH}. "
            "Run `python scripts/select_best_regression_run.py` first."
        )
    
    with open(BEST_MODEL_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    model_uri = config["regression"]["model_uri"]
    model = mlflow.sklearn.load_model(model_uri)
    
    info = {
        "source": "best_model_yaml",
        "model_uri": model_uri,
        "run_id": config["regression"]["run_id"],
        "metrics": {config["regression"]["metric"]: config["regression"]["value"]},
    }
    
    return model, info


# FastAPI App
app = FastAPI(
    title="Amazon Sales Predictor API",
    description="Predict TotalAmount for Amazon sales orders. Supports champion/challenger model selection.",
    version="1.0.0",
)

# Global state
current_model = None
current_model_info: Dict[str, Any] = {}
registry_config: Dict[str, Any] = {}


@app.on_event("startup")
def startup_event():
    """Load the default model when the API starts."""
    global current_model, current_model_info, registry_config
    
    # In Docker mode, load from exported pickle file
    if DOCKER_MODE or EXPORTED_MODEL_PATH.exists():
        try:
            current_model, current_model_info = load_exported_model()
            print(f"Model loaded from exported pickle!")
            print(f"Run ID: {current_model_info.get('run_id', 'N/A')}")
            print(f"Metrics: {current_model_info.get('metrics', {})}")
            return
        except Exception as e:
            print(f"Could not load exported model: {e}")
            if DOCKER_MODE:
                raise  # In Docker mode, model must have exported
    
    registry_config = load_registry_config()
    model_name = registry_config.get("registry", {}).get("regression_model_name")
    
    # Load champion model from registry first
    if model_name:
        try:
            current_model, info = load_model_by_alias(model_name, "champion")
            if info:
                current_model_info = {
                    "source": "registry",
                    "alias": "champion",
                    **info
                }
                print("-" * 70)
                print(f"Champion model loaded from registry!")
                print(f"Model: {model_name} v{info['version']}")
                print(f"Metrics: {info['metrics']}")
                print("-" * 70)
                return
        except Exception as e:
            print(f"Could not load champion from registry: {e}")
    
    # Fallback to best_model.yaml
    try:
        current_model, current_model_info = load_fallback_model()
        current_model_info["source"] = "best_model_yaml"
        print(f"Model loaded from best_model.yaml (fallback)")
        print(f"Run ID: {current_model_info.get('run_id', 'N/A')}")
    except Exception as e:
        print(f"Failed to load any model: {e}")
        raise


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models", response_model=AvailableModelsResponse)
def get_available_models():
    """List all available models (champion, challenger, fallback)."""
    model_name = registry_config.get("registry", {}).get("regression_model_name")
    registered_models = []
    
    if model_name:
        # Check champion
        champion_info = get_registered_model_info(model_name, "champion")
        if champion_info:
            registered_models.append({
                "alias": "champion",
                "version": champion_info["version"],
                "run_id": champion_info["run_id"],
                "metrics": champion_info["metrics"],
                "description": champion_info.get("description", ""),
            })
        
        # Check challenger
        challenger_info = get_registered_model_info(model_name, "challenger")
        if challenger_info:
            registered_models.append({
                "alias": "challenger",
                "version": challenger_info["version"],
                "run_id": challenger_info["run_id"],
                "metrics": challenger_info["metrics"],
                "description": challenger_info.get("description", ""),
            })
    
    return AvailableModelsResponse(
        registered_models=registered_models,
        fallback_available=BEST_MODEL_PATH.exists(),
    )


@app.post("/models/load")
def load_model(alias: str = Query(..., description="Model alias: 'champion', 'challenger', or 'fallback'")):
    """Load a specific model by alias."""
    global current_model, current_model_info
    
    model_name = registry_config.get("registry", {}).get("regression_model_name")
    
    if alias == "fallback":
        try:
            current_model, current_model_info = load_fallback_model()
            current_model_info["source"] = "best_model_yaml"
            return {"message": "Fallback model loaded", "info": current_model_info}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load fallback: {e}")
    
    if not model_name:
        raise HTTPException(status_code=400, detail="No model name configured in registry.yaml")
    
    if alias not in ["champion", "challenger"]:
        raise HTTPException(status_code=400, detail="Alias must be 'champion', 'challenger', or 'fallback'")
    
    try:
        current_model, info = load_model_by_alias(model_name, alias)
        if info:
            current_model_info = {
                "source": "registry",
                "alias": alias,
                **info
            }
            return {"message": f"{alias.capitalize()} model loaded", "info": current_model_info}
        else:
            raise HTTPException(status_code=404, detail=f"Model with alias '{alias}' not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


@app.get("/model-info")
def get_model_info():
    """Get information about the currently loaded model."""
    if not current_model_info:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return current_model_info


@app.post("/predict", response_model=PredictionResponse)
def predict(data: SaleInput):
    """Predict TotalAmount for a single sales order."""
    if current_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    input_df = pd.DataFrame([data.model_dump()])
    prediction = current_model.predict(input_df)[0]
    
    return PredictionResponse(predicted_total_amount=float(prediction))


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(data: List[SaleInput]):
    """Predict TotalAmount for multiple sales orders at once."""
    if current_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty input list")
    
    input_df = pd.DataFrame([item.model_dump() for item in data])
    predictions = current_model.predict(input_df).tolist()
    
    return BatchPredictionResponse(predictions=predictions)
