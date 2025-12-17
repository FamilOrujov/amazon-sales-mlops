from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import List, Optional


# make sure src/ is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))


from amazon_sales_ml.models.train import train_model_with_mlflow   # noqa: E402


DEFAULT_MODELS = [
    "linear_regression",
    "random_forest",
    "hist_gb",
    "xgboost",
    # "lightgbm",  # I decided to not use lightgbm because it was too slow to train
]



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train multiple models and log each run to MLflow."
    )
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Models to train. If omitted, trains all default models.",
    )
    p.add_argument(
        "--no-grid-search",
        action="store_true",
        help="Skip GridSearchCV and fit once per model.",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately if any model training fails.",
    )
    return p.parse_args()


def run(models: List[str], use_grid_search: bool, fail_fast: bool = False) -> None:
    total = len(models)
    print(f"Training {total} model(s). grid_search={use_grid_search}")
    print("-" * 70)

    for i, model_name in enumerate(models, start=1):
        overall_pct = ((i - 1) / total) * 100.0
        print(f"[{i}/{total}] ({overall_pct:5.1f}%) START  {model_name}")

        t0 = time.perf_counter()  # high-resolution timer 
        try:
            result = train_model_with_mlflow(
                model_name=model_name,
                use_grid_search=use_grid_search,
            )

            elapsed = time.perf_counter() - t0
            done_pct = (i / total) * 100.0

            print(
                f"[{i}/{total}] ({done_pct:5.1f}%) DONE   {model_name}  "
                f"in {elapsed:.1f}s  "
                f"test_RMSE={result.metrics_test['RMSE']:.6f}  "
                f"test_R2={result.metrics_test['R2']:.6f}"
            )

        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"[{i}/{total}] ERROR  {model_name}  after {elapsed:.1f}s  {e!r}")
            if fail_fast:
                raise

        print("-" * 70)

    print("All models finished.")
    print("Open MLflow UI with: uv run mlflow ui")




def main() -> None:
    args = parse_args()
    models = args.models if args.models else DEFAULT_MODELS
    use_grid_search = not args.no_grid_search
    run(models=models, use_grid_search=use_grid_search)


if __name__ == "__main__":
    main()

