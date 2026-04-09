import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://fastapi:8000")

st.set_page_config(
    page_title="Stock Market Kubeflow",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Stock Kubeflow")
    st.markdown("---")

    stock = st.selectbox(
        "Stock",
        ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META",
         "RELIANCE.BSE", "TCS.BSE", "INFY", "HDB"],
    )
    model = st.selectbox(
        "Model",
        ["xgboost", "random_forest", "gradient_boosting"],
    )
    days = st.slider("History (days)", 30, 365, 90, step=10)

    st.markdown("---")
    st.caption("Powered by Kubeflow + FastAPI")

# ── Helpers ───────────────────────────────────────────────────────────────────

def api_get(path: str, params: dict = None, timeout: int = 60):
    try:
        r = requests.get(f"{FASTAPI_URL}{path}", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach FastAPI service. Is it running?"
    except requests.exceptions.HTTPError as e:
        return None, f"API error {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return None, str(e)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_history, tab_predict, tab_models, tab_pipeline = st.tabs(
    ["📊 Price History", "🔮 Prediction", "🏆 Model Metrics", "⚙️ Pipeline"]
)

# ────────────────────────── 1. Price History ──────────────────────────────────
with tab_history:
    st.subheader(f"{stock} — Last {days} Days")

    data, err = api_get(f"/history/{stock}", params={"days": days}, timeout=30)
    if err:
        st.error(err)
    elif data:
        df = pd.DataFrame(data["history"])
        df["date"] = pd.to_datetime(df["date"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["close"],
            mode="lines",
            name="Close",
            line=dict(color="#00d4aa", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,212,170,0.08)",
        ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price (USD / INR)",
            template="plotly_dark",
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latest Close", f"{df['close'].iloc[-1]:.2f}")
        col2.metric("Period High", f"{df['close'].max():.2f}")
        col3.metric("Period Low", f"{df['close'].min():.2f}")
        pct = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100
        col4.metric("Period Return", f"{pct:.1f}%", delta=f"{pct:.1f}%")

# ────────────────────────── 2. Prediction ─────────────────────────────────────
with tab_predict:
    st.subheader("Next-Day AAPL Close Prediction")
    st.info(
        "The models were trained on AAPL_Close as the target variable. "
        "Prediction fetches live data, re-runs feature engineering, and infers."
    )

    if st.button("Run Prediction", type="primary"):
        with st.spinner("Fetching live data and computing features…"):
            result, err = api_get("/predict", params={"model": model, "stock": stock})

        if err:
            st.error(err)
        elif result:
            c1, c2, c3 = st.columns(3)
            c1.metric("Stock", result["stock"])
            c2.metric("Predicted Close ($)", f"{result['predicted_close']:.2f}")
            c3.metric("Model", result["model"].replace("_", " ").title())
            st.success(f"Prediction for **{result['prediction_date']}** complete.")

# ────────────────────────── 3. Model Metrics ──────────────────────────────────
with tab_models:
    st.subheader("Trained Model Comparison")

    models_data, err = api_get("/models", timeout=10)
    if err:
        st.error(err)
    elif models_data:
        st.write("**Available models:**", ", ".join(models_data.get("available", [])))

        best = models_data.get("best_model")
        if best:
            st.success(
                f"Best model: **{best.get('Model')}** — "
                f"MSE={best.get('MSE', 0):.4f}  "
                f"RMSE={best.get('RMSE', 0):.4f}  "
                f"R²={best.get('R2', 0):.4f}"
            )
        else:
            st.info("Run the full pipeline to generate model comparison metrics.")

    # Try to load full results CSV from a mounted volume for rich chart
    results_path = os.path.join(
        os.environ.get("DATA_DIR", "/data"), "models", "all_models_results.csv"
    )
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        st.markdown("### Metrics Table")
        st.dataframe(results_df, use_container_width=True)

        fig = px.bar(
            results_df.melt(id_vars="Model", value_vars=["MSE", "RMSE", "R2"]),
            x="Model", y="value", color="variable", barmode="group",
            template="plotly_dark",
            title="Model Metrics Comparison",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("(Full metrics chart available after pipeline run)")

# ────────────────────────── 4. Pipeline ───────────────────────────────────────
with tab_pipeline:
    st.subheader("Kubeflow Pipeline Status")

    health, err = api_get("/health", timeout=5)
    if err:
        st.error(f"FastAPI: {err}")
    else:
        st.success("FastAPI service is **online**")
        loaded = health.get("models_loaded", [])
        if loaded:
            st.write("Loaded models:", ", ".join(loaded))
        else:
            st.warning("No models loaded — run the training pipeline first.")

    st.markdown("---")
    st.markdown("### Pipeline Stages")

    stages = [
        ("1", "Data Fetching", "fetch_stock_data.py",
         "Downloads 10 years OHLCV data for 10 stocks via yfinance"),
        ("2", "Feature Engineering", "feature_engineering.py",
         "Computes MA20, MA50, RSI(14), MACD for each stock"),
        ("3a", "XGBoost Training", "train_xgboost.py",
         "XGBRegressor — 300 estimators, lr=0.05, max_depth=6"),
        ("3b", "Gradient Boosting", "train_gradient_boosting.py",
         "GradientBoostingRegressor — 200 estimators, lr=0.05, max_depth=5"),
        ("3c", "Random Forest", "train_random_forest.py",
         "RandomForestRegressor — 200 trees, max_depth=10"),
        ("4", "Model Evaluation", "model_evaluation.py",
         "Evaluates MSE / RMSE / R², picks best model by MSE"),
    ]

    for num, name, script, desc in stages:
        with st.expander(f"Stage {num}: {name}"):
            st.code(script, language="bash")
            st.caption(desc)
