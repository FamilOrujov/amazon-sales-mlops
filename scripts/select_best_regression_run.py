from __future__ import annotations

from pathlib import Path
import yaml
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
OUT_PATH = PROJECT_ROOT / "configs" / "best_model.yaml"


def main() -> None:
    with CFG_PATH.open("r") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg["mlflow"]["experiment_name"]
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        raise RuntimeError(f"MLflow experiment not found: {exp_name}")
    

    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    if runs.empty:
        raise RuntimeError(f"No runs found in experiment: {exp_name}")
    
    
    best = runs.sort_values("metrics.test_RMSE", ascending=True).iloc[0]
    best_run_id = best["run_id"]
    best_model_uri = f"runs:/{best_run_id}/model"

    OUT_PATH.write_text(
        yaml.safe_dump(
            {
                "regression": {
                    "run_id": best_run_id,
                    "model_uri": best_model_uri,
                    "metric": "test_RMSE",
                    "value": float(best["metrics.test_RMSE"]),
                }
            },
            sort_keys=False,
        )
    )

    print("Saved best model:")
    print(f"  run_id:   {best_run_id}")
    print(f"  model_uri:{best_model_uri}")


if __name__ == "__main__":
    main()

