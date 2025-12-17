from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PathsConfig:
    processed_data: Path


@dataclass
class TrainingConfig:
    target_col: str
    test_size: float
    random_state: int


@dataclass
class MLflowConfig:
    experiment_name: str
    tracking_uri: Optional[str]


@dataclass
class AppConfig:
    paths: PathsConfig
    training: TrainingConfig
    mlflow: MLflowConfig


def load_config(config_path: Path | None = None) -> AppConfig:
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"

    with config_path.open("r") as f:
        raw = yaml.safe_load(f)

    paths = PathsConfig(
        processed_data=PROJECT_ROOT / raw["paths"]["processed_data"],
    )

    training = TrainingConfig(
        target_col=raw["training"]["target_col"],
        test_size=float(raw["training"]["test_size"]),
        random_state=int(raw["training"]["random_state"]),
    )

    mlflow_cfg = MLflowConfig(
        experiment_name=raw["mlflow"]["experiment_name"],
        tracking_uri=raw["mlflow"]["tracking_uri"],
    )

    return AppConfig(
        paths=paths,
        training=training,
        mlflow=mlflow_cfg
    )

