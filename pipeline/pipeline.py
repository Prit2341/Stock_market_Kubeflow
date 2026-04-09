"""
Stock Market Prediction — Kubeflow Pipeline

DAG:
  fetch_data
      |
  feature_engineering
      |
  train_xgboost + train_gb + train_rf  (parallel)
      |
  model_evaluation

All steps share a single PVC mounted at /data.
Images must be loaded into k3d before running:
    pipeline\\build_images.ps1
"""

from kfp import dsl, compiler
from kfp.kubernetes import mount_pvc, set_image_pull_policy, use_secret_as_env

# ── Image names (loaded into k3d via build_images.ps1) ──────────────────────
IMAGES = {
    "fetch":    "stock-fetch-data:latest",
    "features": "stock-feature-engineering:latest",
    "xgb":      "stock-train-xgboost:latest",
    "gb":       "stock-train-gradient-boosting:latest",
    "rf":       "stock-train-random-forest:latest",
    "eval":     "stock-model-evaluation:latest",
}

PVC_SIZE  = "5Gi"
DATA_DIR  = "/data"   # must match os.environ.get("DATA_DIR", "/data") in scripts


# ── Component definitions ────────────────────────────────────────────────────
# Scripts default to DATA_DIR=/data and MODELS_DIR=/data/models via os.environ.get(),
# which matches the PVC mount point — no need to pass env vars explicitly.

@dsl.container_component
def fetch_data():
    return dsl.ContainerSpec(
        image=IMAGES["fetch"],
        command=["python", "src/fetch_stock_data.py"],
    )


@dsl.container_component
def feature_engineering():
    return dsl.ContainerSpec(
        image=IMAGES["features"],
        command=["python", "src/feature_engineering.py"],
    )


@dsl.container_component
def train_xgboost():
    return dsl.ContainerSpec(
        image=IMAGES["xgb"],
        command=["python", "src/train_xgboost.py"],
    )


@dsl.container_component
def train_gradient_boosting():
    return dsl.ContainerSpec(
        image=IMAGES["gb"],
        command=["python", "src/train_gradient_boosting.py"],
    )


@dsl.container_component
def train_random_forest():
    return dsl.ContainerSpec(
        image=IMAGES["rf"],
        command=["python", "src/train_random_forest.py"],
    )


@dsl.container_component
def model_evaluation():
    return dsl.ContainerSpec(
        image=IMAGES["eval"],
        command=["python", "src/model_evaluation.py"],
    )


# ── Pipeline definition ──────────────────────────────────────────────────────

@dsl.pipeline(
    name="stock-prediction-pipeline",
    description="End-to-end stock market ML pipeline: fetch > features > train x3 > evaluate",
)
def stock_prediction_pipeline():

    # ── Use persistent pre-populated PVC ─────────────────────────────────────
    # PVC is created once via: kubectl apply -f pipeline/stock-data-pvc.yaml
    # Data is loaded via: pipeline\prepare_data.ps1
    PVC_NAME = "stock-data"

    def _mount(task):
        """Mount the shared PVC, set pull policy, disable caching."""
        mount_pvc(task, pvc_name=PVC_NAME, mount_path=DATA_DIR)
        set_image_pull_policy(task, "IfNotPresent")
        task.set_caching_options(enable_caching=False)
        return task

    # Step 1 — fetch stock data via Alpha Vantage API
    fetch_task = _mount(fetch_data())
    use_secret_as_env(fetch_task, secret_name="alpha-vantage",
                      secret_key_to_env={"api-key": "ALPHA_VANTAGE_API_KEY"})

    # Step 2 — Feature engineering (RSI, MACD, SMA, EMA …)
    features_task = _mount(feature_engineering().after(fetch_task))

    # Step 3 — Train three models in parallel
    xgb_task = _mount(train_xgboost().after(features_task))
    gb_task  = _mount(train_gradient_boosting().after(features_task))
    rf_task  = _mount(train_random_forest().after(features_task))

    # Step 4 — Pick best model, write results CSV
    _mount(model_evaluation().after(xgb_task, gb_task, rf_task))


# ── Compile ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    output = "pipeline.yaml"
    compiler.Compiler().compile(stock_prediction_pipeline, output)
    print(f"Pipeline compiled → {output}")
