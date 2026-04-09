# Stock Market Prediction — Kubeflow MLOps Pipeline

A production-grade ML pipeline for stock price prediction using Kubeflow Pipelines, deployed on a 2-node k3s cluster connected via Tailscale VPN, with a full serving layer and Jenkins-triggered CI/CD.

---

## Architecture

```
Jenkins (slave) → submit_pipeline.py → Kubeflow Pipelines (master)
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         │                         │
             fetch-data              feature-engineering         (parallel training)
                    │                         │
          ┌─────────┴──────────┐    train-gradient-boosting
          │                    │    train-random-forest
     combined CSV         features CSV    train-xgboost
          │                    │              │
          └─────────┬──────────┘    model-evaluation
                    │
              PVC (stock-data)
                    │
            Docker Compose (serving)
          ┌─────────┼─────────┐
        FastAPI  Streamlit  Grafana
                    │
               Prometheus
```

---

## Cluster Setup

| Node | Role | IP (Tailscale) |
|------|------|----------------|
| Desktop (master) | k3s server + Kubeflow | 100.76.106.42 |
| Laptop (slave) | k3s agent + Jenkins | 100.103.66.18 |

---

## Stocks Tracked

| Ticker | Stock |
|--------|-------|
| AAPL | Apple |
| MSFT | Microsoft |
| GOOGL | Google |
| AMZN | Amazon |
| TSLA | Tesla |
| META | Meta |
| RELIANCE.BSE | Reliance Industries |
| TCS.BSE | Tata Consultancy Services |
| INFY | Infosys |
| HDB | HDFC Bank |

---

## ML Models

- **XGBoost** — gradient boosted trees
- **Random Forest** — ensemble method
- **Gradient Boosting** — sklearn GradientBoostingRegressor

**Features:** Close price, MA10, MA20, RSI, MACD, Signal, Volume  
**Training data:** 10 years historical data, 80/20 train/test split

---

## Services

| Service | Port | URL |
|---------|------|-----|
| Kubeflow UI | 8080 | http://localhost:8080 |
| FastAPI | 8000 | http://localhost:8000 |
| Streamlit | 8501 | http://localhost:8501 |
| Grafana | 3001 | http://localhost:3001 |
| Prometheus | 9090 | http://localhost:9090 |

---

## Running the Pipeline

**Manual trigger:**
```powershell
cd pipeline
python submit_pipeline.py
```

**Automated:** Jenkins job on slave triggers daily at market close (4 PM).

---

## Serving Layer

```powershell
# Start all services
docker compose up -d

# Check status
docker compose ps
```

---

## Kubeflow Port-Forward (for remote access)

```powershell
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80 --address 0.0.0.0
```

---

## Project Structure

```
Stock_Kubeflow/
├── pipeline/
│   ├── pipeline.py          # Kubeflow pipeline definition
│   ├── submit_pipeline.py   # Trigger pipeline run
│   └── run_pipeline.ps1     # PowerShell helper
├── components/
│   ├── 1_fetch_data/        # Data ingestion component
│   ├── 2_feature_engineering/
│   ├── 3_train_gradient_boosting/
│   ├── 3_train_random_forest/
│   ├── 3_train_xgboost/
│   └── 4_model_evaluation/
├── services/
│   ├── fastapi/             # Prediction API
│   ├── streamlit/           # Dashboard UI
│   ├── prometheus/          # Metrics collection
│   └── grafana/             # Metrics visualization
├── data/                    # Exported from Kubeflow PVC (gitignored)
├── docker-compose.yml
└── PROJECT_REPORT.md
```

---

## Key Fixes & Lessons Learned

- `kubectl cp` with Windows drive letters (`d:\`) fails — use relative paths
- `yfinance` is blocked inside Docker containers — read from local CSV instead
- `python:3.11-slim` has no `curl` — use Python urllib for healthchecks
- Kubeflow port-forward needs `--address 0.0.0.0` for remote access
- Indian stock tickers use `.BSE` suffix on Alpha Vantage, not `.NS`
- pip hash mismatch during Docker build — add `pip install --upgrade pip` first
