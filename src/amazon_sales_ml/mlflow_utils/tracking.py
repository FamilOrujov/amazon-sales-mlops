from __future__ import annotations

from typing import Dict, Any, Optional

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


def init_mlflow(experiment_name: str, tracking_uri: Optional[str] = None) -> None:
    
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)


def log_model_run(
    model,
    model_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    X_sample,
    y_sample=None,
    run_name: Optional[str] = None,
) -> str:
    
    if run_name is None:
        run_name = model_name

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("model_name", model_name)

        # Log hyperparameters
        if params:
            mlflow.log_params(params)

        # Log metrics (train_* and test_* values)
        if metrics:
            mlflow.log_metrics(metrics)

        # Infer model signature from sample data
        signature = infer_signature(X_sample, model.predict(X_sample))
        
        # Use first row as input example (helps with model serving)
        input_example = X_sample.head(1)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
        )
        
        return run.info.run_id

