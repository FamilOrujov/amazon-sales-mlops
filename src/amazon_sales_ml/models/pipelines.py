from __future__ import annotations

from typing import Dict, Any, List

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor


try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None


# Prepocessor builders

def _make_linear_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Preprocesser for linear models"""
    cat_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
    )

    num_scaler = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_scaler, numeric_features),
            ("cat", cat_encoder, categorical_features),
        ]
    )
    return preprocessor


def _make_tree_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Preprocesser for tree-based models"""
    cat_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )


    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", cat_encoder, categorical_features)
        ],
        sparse_threshold=0  # Force dense output for models like HistGradientBoosting
    )
    return preprocessor


def _to_dense(X):
    """Ensure dense arrays for estimators that do not accept sparse input."""
    return X.toarray() if hasattr(X, "toarray") else X


# Model factory

def _build_estimator(model_name: str):
    model_name = model_name.lower()

    if model_name == "linear_regression":
        return LinearRegression()

    if model_name == "random_forest":
        return RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
        )

    if model_name == "hist_gb":
        return HistGradientBoostingRegressor(
            random_state=42,
        )

    if model_name == "xgboost":
        if XGBRegressor is None:
            raise ImportError(
                "xgboost is not installed. Install it with `uv ad xgboost`."
            )
        return XGBRegressor(
            tree_method="hist",
            random_state=42,
            n_estimators=3000,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
        )

    if model_name == "lightgbm":
        if LGBMRegressor is None:
            raise ImportError(
                "lightgbm is not installed. Install it with `uv ad lightgbm`."
            )
        return LGBMRegressor(
            random_state=42,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
            n_jobs=-1,
        )
    
    raise ValueError(f"Unknown model_name: {model_name!r}")


# Public API -> pipelines + param grids

def get_pipeline(
    model_name: str,
    numeric_features: List[str],
    categorical_features: List[str],
) -> Pipeline:
    """Get a pipeline for a given model name"""
    model_name_lower = model_name.lower()

    if model_name_lower == "linear_regression":
        preprocessor = _make_linear_preprocessor(numeric_features, categorical_features)
    else:
        # Tree-based models (RF, HistGB, XGBoost, LightGBM)
        preprocessor = _make_tree_preprocessor(numeric_features, categorical_features)
    
    estimator = _build_estimator(model_name_lower)

    steps = [("preprocessor", preprocessor)]

    if model_name_lower == "hist_gb":
        steps.append(("to_dense", FunctionTransformer(_to_dense, accept_sparse=True)))

    steps.append(("model", estimator))

    pipeline = Pipeline(steps=steps)
    return pipeline


def get_param_grid(model_name: str) -> Dict[str, List[Any]]:
    """Get a parameter grid for a given model name"""
    model_name = model_name.lower()

    if model_name == "linear_regression":
        # No hyperparameters for plain LinearRegression
        return {}

    if model_name == "random_forest":
        return {
            "model__n_estimators": [100],        #[100, 200]
            "model__max_depth": [None, 10],      #[None, 10, 20]
            "model__min_samples_split": [2],     #[2, 5]
            "model__min_samples_leaf": [1],      #[1, 2]
            "model__max_features": ["sqrt"],     #[sqrt, log2]
        }

    if model_name == "hist_gb":
        return {
            "model__max_depth": [None, 6, 10],
            "model__learning_rate": [0.05, 0.1],
            "model__max_iter": [200, 400],
            "model__min_samples_leaf": [20, 50],
        }

    if model_name == "xgboost":
        return {
            "model__n_estimators": [200, 400],
            "model__max_depth": [4, 6],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        }

    if model_name == "lightgbm":
        return {
          
            "model__n_estimators": [200, 300],
            "model__num_leaves": [31, 63],
            "model__learning_rate": [0.05],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8],
        }

    raise ValueError(f"Unknown model_name for param grid: {model_name!r}")
