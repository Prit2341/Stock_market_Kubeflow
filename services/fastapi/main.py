import os
import time
import logging
from datetime import datetime, timedelta
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import ta
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("DATA_DIR", "/data")
MODELS_DIR = os.environ.get("MODELS_DIR", "/data/models")

STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META",
    "RELIANCE.BSE", "TCS.BSE", "INFY", "HDB",
]

MODEL_FILES = {
    "xgboost": "XGBoost_model.pkl",
    "random_forest": "random_forest_model.pkl",
    "gradient_boosting": "gradient_boosting_model.pkl",
}

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds", "API request latency in seconds", ["endpoint"]
)
PREDICTION_COUNT = Counter(
    "predictions_total", "Total predictions made", ["model", "stock"]
)
MODEL_MSE = Gauge("model_mse", "Model MSE from evaluation", ["model"])
MODEL_RMSE = Gauge("model_rmse", "Model RMSE from evaluation", ["model"])
MODEL_R2 = Gauge("model_r2", "Model R2 score from evaluation", ["model"])

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Stock Market Prediction API",
    description="Serves predictions from the Kubeflow-trained models",
    version="1.0.0",
)

models: dict = {}


def load_models() -> None:
    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)
            logger.info(f"Loaded model: {name}")
        else:
            logger.warning(f"Model file not found: {path}")

    # Populate Prometheus gauges from evaluation results
    results_csv = os.path.join(MODELS_DIR, "all_models_results.csv")
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        for _, row in df.iterrows():
            label = row["Model"].lower().replace(" ", "_")
            MODEL_MSE.labels(model=label).set(row.get("MSE", 0))
            MODEL_RMSE.labels(model=label).set(row.get("RMSE", 0))
            MODEL_R2.labels(model=label).set(row.get("R2", 0))


def _flatten_columns(df: pd.DataFrame, stock: str) -> pd.DataFrame:
    """Handle both flat and MultiIndex yfinance column layouts."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.rename(
        columns={
            "Open": f"{stock}_Open",
            "High": f"{stock}_High",
            "Low": f"{stock}_Low",
            "Close": f"{stock}_Close",
            "Adj Close": f"{stock}_AdjClose",
            "Volume": f"{stock}_Volume",
        },
        inplace=True,
    )
    return df


def build_live_features() -> pd.DataFrame:
    """Load pre-computed features from the pipeline output."""
    features_path = os.path.join(DATA_DIR, "features", "stock_features.csv")
    if not os.path.exists(features_path):
        raise RuntimeError(f"Features file not found at {features_path}. Run the pipeline first.")
    df = pd.read_csv(features_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ── Events ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event() -> None:
    load_models()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "models_loaded": list(models.keys())}


@app.get("/metrics", tags=["ops"])
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/models", tags=["models"])
def list_models():
    available = list(models.keys())
    best_path = os.path.join(MODELS_DIR, "best_model.csv")
    best = None
    if os.path.exists(best_path):
        best = pd.read_csv(best_path).to_dict(orient="records")[0]
    return {"available": available, "best_model": best}


@app.get("/predict", tags=["inference"])
def predict(
    model: Literal["xgboost", "random_forest", "gradient_boosting"] = "xgboost",
    stock: str = "AAPL",
):
    t0 = time.time()
    if model not in models:
        REQUEST_COUNT.labels(method="GET", endpoint="/predict", status="404").inc()
        raise HTTPException(
            404, f"Model '{model}' not loaded. Available: {list(models.keys())}"
        )

    try:
        df = build_live_features()
        feature_row = df.drop(columns=["Date", "AAPL_Close"], errors="ignore").tail(1)
        prediction = float(models[model].predict(feature_row)[0])

        PREDICTION_COUNT.labels(model=model, stock=stock).inc()
        REQUEST_COUNT.labels(method="GET", endpoint="/predict", status="200").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - t0)

        return {
            "stock": stock,
            "model": model,
            "predicted_close": round(prediction, 2),
            "prediction_date": datetime.today().strftime("%Y-%m-%d"),
        }
    except Exception as exc:
        REQUEST_COUNT.labels(method="GET", endpoint="/predict", status="500").inc()
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(500, str(exc))


@app.get("/history/{stock}", tags=["data"])
def history(stock: str, days: int = 90):
    t0 = time.time()
    try:
        combined_path = os.path.join(DATA_DIR, "combined", "all_10_stocks_10years.csv")
        if not os.path.exists(combined_path):
            raise HTTPException(404, "Combined data file not found. Run the pipeline first.")

        df = pd.read_csv(combined_path, parse_dates=["Date"])
        close_col = f"{stock}_Close"
        if close_col not in df.columns:
            raise HTTPException(404, f"No data found for {stock}")

        cutoff = df["Date"].max() - timedelta(days=days)
        df = df[df["Date"] >= cutoff][["Date", close_col]].dropna()
        records = df.rename(columns={"Date": "date", close_col: "close"})
        records["date"] = records["date"].astype(str)

        REQUEST_COUNT.labels(method="GET", endpoint="/history", status="200").inc()
        REQUEST_LATENCY.labels(endpoint="/history").observe(time.time() - t0)
        return {"stock": stock, "days": days, "history": records.to_dict(orient="records")}

    except HTTPException:
        raise
    except Exception as exc:
        REQUEST_COUNT.labels(method="GET", endpoint="/history", status="500").inc()
        raise HTTPException(500, str(exc))
