import pickle
from pathlib import Path

import yaml
import mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEST_MODEL_PATH = PROJECT_ROOT / "configs" / "best_model.yaml"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_PATH = MODELS_DIR / "champion.pkl"


def main():
    # Create models directory
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Load best model config
    with open(BEST_MODEL_PATH) as f:
        config = yaml.safe_load(f)
    
    model_uri = config["regression"]["model_uri"]
    
    print(f"Loading model from: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    
    # Save model info alongside
    model_data = {
        "model": model,
        "info": {
            "run_id": config["regression"]["run_id"],
            "metric": config["regression"]["metric"],
            "metric_value": config["regression"]["value"],
        }
    }
    
    # Export to pickle
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Model exported to: {OUTPUT_PATH}")
    print(f"   Run ID: {config['regression']['run_id']}")
    print(f"   {config['regression']['metric']}: {config['regression']['value']:.4f}")


if __name__ == "__main__":
    main()

