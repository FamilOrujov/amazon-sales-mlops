from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from mlflow import MlflowClient
import pandas as pd
import yaml



PROJECT_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = PROJECT_ROOT / "configs" / "registry.yaml"

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT)
        return out.decode().strip()
    except Exception:
        return None
    


@dataclass
class Gates:
    min_test_r2: float
    max_test_rmse: float
    max_generalization_gap_pct: float


@dataclass
class RegistrySpec:
    experiment_name: str
    tracking_uri: Optional[str]
    regression_model_name: str
    register_top_k: int
    champion_alias: str
    challenger_alias: str
    gates: Gates
    dataset_id: str
    data_path: Path


def load_spec() -> RegistrySpec:
    raw = yaml.safe_load(CFG_PATH.read_text())

    tracking_uri = raw["mlflow"].get("tracking_uri")
    exp_name = raw["mlflow"]["experiment_name"]

    reg = raw["registry"]
    gates = Gates(
        min_test_r2=float(reg["min_test_r2"]),
        max_test_rmse=float(reg["max_test_rmse"]),
        max_generalization_gap_pct=float(reg["max_generalization_gap_pct"]),
    )

    meta = raw.get("metadata", {})
    data_path = PROJECT_ROOT / meta.get("data_path", "data/processed/amazon_sales_regression.csv")

    return RegistrySpec(
        experiment_name=exp_name,
        tracking_uri=tracking_uri,
        regression_model_name=reg["regression_model_name"],
        register_top_k=int(reg["register_top_k"]),
        champion_alias=reg["champion_alias"],
        challenger_alias=reg["challenger_alias"],
        gates=gates,
        dataset_id=str(meta.get("dataset_id", "unknown_dataset")),
        data_path=data_path,
    )


def require_metrics(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            "Missing expected metrics in MLflow runs: "
            + ", ".join(missing)
            + ". Make sure you log these metrics in train_model_with_mlflow."
        )
    

def filter_and_rank_runs(df: pd.DataFrame, gates: Gates) -> pd.DataFrame:
    # Require columns (train_RMSE is optional for gap check)
    require_metrics(
        df,
        [
            "run_id",
            "metrics.test_R2",
            "metrics.test_RMSE",
        ],
    )

    # Calculate overfitting gap (if train_RMSE exists)
    # If train_RMSE is missing (NaN), set gap to 0 (skip overfitting check)
    if "metrics.train_RMSE" in df.columns:
        gap_pct = (df["metrics.test_RMSE"] - df["metrics.train_RMSE"]) / df["metrics.train_RMSE"] * 100.0
        # Fill NaN gaps with 0 (passes the check when train_RMSE is missing)
        gap_pct = gap_pct.fillna(0)
    else:
        gap_pct = 0
    
    df = df.assign(_gap_pct=gap_pct)

    # Apply gates
    ok = df[
        (df["metrics.test_R2"] >= gates.min_test_r2)
        & (df["metrics.test_RMSE"] <= gates.max_test_rmse)
        & (df["_gap_pct"] <= gates.max_generalization_gap_pct)
    ].copy()

    # Rank: highest R2, then lowest RMSE
    ok = ok.sort_values(
        by=["metrics.test_R2", "metrics.test_RMSE"],
        ascending=[False, True],
    )
    return ok


def register_version(
    client: MlflowClient,
    model_uri: str,
    model_name: str,
    tags: Dict[str, str],
    description: str,
) -> int:
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = int(mv.version)

    for k, v in tags.items():
        client.set_model_version_tag(name=model_name, version=str(version), key=k, value=v)

    # Readable description for the version
    client.update_model_version(name=model_name, version=str(version), description=description)

    return version


def set_alias_safe(client: MlflowClient, model_name: str, alias: str, version: int) -> None:
    client.set_registered_model_alias(model_name, alias, str(version))


def main() -> None:
    spec = load_spec()

    if spec.tracking_uri:
        mlflow.set_tracking_uri(spec.tracking_uri)

    exp = mlflow.get_experiment_by_name(spec.experiment_name)
    if exp is None:
        raise RuntimeWarning(f"Experiment not found: {spec.experiment_name}")
    
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    if runs.empty:
        raise RuntimeError(f"No runs found in experiment: {spec.experiment_name}")
    
    qualified = filter_and_rank_runs(runs, spec.gates)
    if qualified.empty:
        print("No runs passed the registration gates.")
        print("Tip. relax thresholds in configs/registry.yaml or confirm metrics are logged")
        return
    
    top = qualified.head(spec.register_top_k).reset_index(drop=True)

    client = MlflowClient()

    git_commit = get_git_commit()
    data_hash = sha256_file(spec.data_path) if spec.data_path.exists() else "missing_file"

    print(f"Qualified runs: {len(qualified)}. Registering top {len(top)}.")
    print("-" * 80)

    registered_versions: list[int] =[]

    for idx, row in top.iterrows():
        run_id = row["run_id"]
        model_uri = f"runs:/{run_id}/model"

        test_r2 = float(row["metrics.test_R2"])
        test_rmse = float(row["metrics.test_RMSE"])
        gap_pct = float(row["_gap_pct"])

        tags = {
            "task": "regression",
            "target": "TotalAmount",
            "dataset_id": spec.dataset_id,
            "data_sha256": data_hash,
            "source_run_id": run_id,
            "git_commit": git_commit or "unknown",
        }

        desc = (
            f"Auto-registered from experiment='{spec.experiment_name}'. "
            f"test_R2={test_r2:.6f}, test_RMSE={test_rmse:.6f}, gap_pct={gap_pct:.2f}. "
            f"source_run_id={run_id}."
        )

        version = register_version(
            client=client,
            model_uri=model_uri,
            model_name=spec.regression_model_name,
            tags=tags,
            description=desc,
        )
        registered_versions.append(version)

        print(
            f"[{idx+1}] registered model='{spec.regression_model_name}' version={version} "
            f"run_id={run_id} test_R2={test_r2:.6f} test_RMSE={test_rmse:.6f} gap%={gap_pct:.2f}"
        )

    # Aliases: champion = best, challenger = second best (if present)
    champion_version = registered_versions[0]
    set_alias_safe(client, spec.regression_model_name, spec.champion_alias, champion_version)

    if len(registered_versions) > 1:
        challenger_version = registered_versions[1]
        set_alias_safe(client, spec.regression_model_name, spec.challenger_alias, challenger_version)

    print("-" * 80)
    print("Aliases set:")
    print(f"  {spec.champion_alias} -> {spec.regression_model_name} v{champion_version}")
    if len(registered_versions) > 1:
        print(f"  {spec.challenger_alias} -> {spec.regression_model_name} v{registered_versions[1]}")

    print("\nYou can load the champion model with:")
    print(f"  mlflow.pyfunc.load_model('models:/{spec.regression_model_name}@{spec.champion_alias}')")


if __name__ == "__main__":
    main()

