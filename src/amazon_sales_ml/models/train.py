from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV

from amazon_sales_ml.config import load_config
from amazon_sales_ml.models.pipelines import get_pipeline, get_param_grid
from amazon_sales_ml.mlflow_utils.tracking import init_mlflow, log_model_run
from amazon_sales_ml.models.evaluate import evaluate_regression 


@dataclass
class TrainResult:
    model_name: str
    best_model: Any
    best_params: Dict[str, Any]
    metrics_train: Dict[str, float]
    metrics_test: Dict[str, float]


def load_processed_dataset(path: Path, target_col: str):
    df = pd.read_csv(path)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    return X, y, numeric_features, categorical_features


def _fit_with_optional_grid_search(
    model_name: str,
    X_train,
    y_train,
    numeric_features,
    categorical_features,
    use_grid_search: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    pipeline = get_pipeline(model_name, numeric_features, categorical_features)

    if not use_grid_search:
        pipeline.fit(X_train, y_train)
        # take underlying mode params for logging
        model_params = pipeline.named_steps["model"].get_params()
        return pipeline, model_params
    
    param_grid = get_param_grid(model_name)
    if not param_grid:
        # no grid, just fit once
        pipeline.fit(X_train, y_train)
        model_params = pipeline.named_steps["model"].get_params()
        return pipeline, model_params
    
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    return best_model, best_params



def train_model(
    model_name: str,
    use_grid_search: bool = True,
) -> TrainResult:
    """Main entry point for training a single model type"""
    cfg = load_config()

    # load data
    X, y, numeric_features, categorical_features = load_processed_dataset(
        cfg.paths.processed_data,
        cfg.training.target_col,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state,
        shuffle=True,
    )

    # fit model
    best_model, best_params = _fit_with_optional_grid_search(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        use_grid_search=use_grid_search,
    )

    # evaluate
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    metrics_train = evaluate_regression(y_train, y_pred_train)
    metrics_test = evaluate_regression(y_test, y_pred_test)

    return TrainResult(
        model_name=model_name,
        best_model=best_model,
        best_params=best_params,
        metrics_train=metrics_train,
        metrics_test=metrics_test,
    )


def train_model_with_mlflow(
    model_name: str,
    use_grid_search: bool = True,
) -> TrainResult:
    """Train the model and log to MLflow in a single call"""
    cfg = load_config()
    init_mlflow(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
    )

    result = train_model(
        model_name=model_name,
        use_grid_search=use_grid_search,
    )

    # prepare metrics to log under single dict
    metrics_to_log = {
        "train_MAE": result.metrics_train["MAE"],
        "train_RMSE": result.metrics_train["RMSE"],
        "train_R2": result.metrics_train["R2"],
        "test_MAE": result.metrics_test["MAE"],
        "test_RMSE": result.metrics_test["RMSE"],
        "test_R2": result.metrics_test["R2"],
    }

    # small sample for signature
    X_sample = pd.read_csv(cfg.paths.processed_data).drop(columns=[cfg.training.target_col]).head(200)

    log_model_run(
        model=result.best_model,
        model_name=model_name,
        params=result.best_params,
        metrics=metrics_to_log,
        X_sample=X_sample,
    )

    return result
