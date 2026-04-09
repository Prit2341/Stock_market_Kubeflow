# Stock Market ML Pipeline on Kubeflow: Complete Technical Report

**Project**: Stock Market Prediction Pipeline using Kubeflow on Kubernetes (k3d)  
**Author**: Student/Developer (MLOps Learning Journey)  
**Environment**: Windows 11, Docker Desktop, k3d, Kubeflow Pipelines 2.16.0  
**Date**: April 2026  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure and Branches](#2-repository-structure-and-branches)
3. [Part 1 — Code Bug Fixes in Pipeline Scripts](#3-part-1--code-bug-fixes-in-pipeline-scripts)
4. [Part 2 — Production Services Built from Scratch](#4-part-2--production-services-built-from-scratch)
5. [Part 3 — Kubeflow Installation Nightmare](#5-part-3--kubeflow-installation-nightmare)
6. [Part 4 — Building the KFP Pipeline](#6-part-4--building-the-kfp-pipeline)
7. [Part 5 — fetch-data Internet Access Issues](#7-part-5--fetch-data-internet-access-issues)
8. [Part 5b — Docker Image Patching Without Internet](#8-part-5b--docker-image-patching-without-internet)
9. [Part 5c — NaN Training Failures](#9-part-5c--nan-training-failures)
10. [Key Concepts Learned](#10-key-concepts-learned)
11. [Summary Table of All Problems and Fixes](#11-summary-table-of-all-problems-and-fixes)
12. [Part 6 — Connecting the Serving Layer](#12-part-6--connecting-the-serving-layer)

---

## 1. Project Overview

This project is a complete, end-to-end Stock Market Machine Learning Pipeline built on top of Kubeflow Pipelines running inside a Kubernetes cluster provisioned via k3d (Kubernetes in Docker). The goal was to take an existing student codebase, fix its bugs, and productionize it into a fully automated ML pipeline with proper monitoring, a REST API, and a user-facing dashboard.

### Technology Stack

| Layer | Tool/Service |
|---|---|
| Containerization | Docker Desktop |
| Kubernetes | k3d (k3s in Docker), cluster named `stockcluster` |
| ML Pipeline Orchestration | Kubeflow Pipelines 2.16.0 |
| Pipeline SDK | kfp==2.10.1, kfp-kubernetes==1.3.0 |
| Data Fetching | Alpha Vantage API (switched from yfinance) |
| ML Models | XGBoost, Gradient Boosting, Random Forest |
| REST API | FastAPI |
| Dashboard | Streamlit |
| Metrics | Prometheus |
| Visualization | Grafana |

### High-Level Architecture

```
[Alpha Vantage API]
        |
        v
[fetch_data] --> [feature_engineering] --> [train_xgboost]
                                       --> [train_gradient_boosting]  --> [model_evaluation]
                                       --> [train_random_forest]
        |
     (PVC: stock-data, mounted at /data on all steps)
        |
        v
[FastAPI :8000] <-- [Streamlit :8501]
        |
        v
[Prometheus :9090] --> [Grafana :3001]
```

The pipeline runs on a Kubeflow-enabled k3d cluster. All pipeline steps share a single PersistentVolumeClaim (PVC) named `stock-data` mounted at `/data`, allowing data to flow from one step to the next without any external object storage.

---

## 2. Repository Structure and Branches

The original repository was cloned in read-only mode from:

```
https://github.com/Prit2341/Stock_market_Kubeflow.git
```

The repo contained 4 branches, each built on top of the previous one:

| Branch | Contribution |
|---|---|
| `vinit` | Data fetching script (`1_fetch_data.py`) |
| `vidhi` | Feature engineering script (`2_feature_engineering.py`) |
| `shruti` | Three model training scripts (`3_train_*.py`) |
| `jay` | Model evaluation script (`4_model_evaluation.py`) |

Each branch added new pipeline components, but all of them contained bugs that prevented them from running in a containerized Kubernetes environment. The goal was to fix all bugs and extend the project with production-grade services.

---

## 3. Part 1 — Code Bug Fixes in Pipeline Scripts

### 3.1 `1_fetch_data.py` — Data Fetching Script

This script downloads historical stock price data. It was written assuming it would run locally on a developer laptop. Running it inside a Docker container on Kubernetes broke it in multiple ways.

---

#### Bug 1: Hardcoded Relative File Paths

**Problem**: The script used hardcoded relative paths like `../data/` to save output files.

```python
# Original (broken in containers)
data.to_csv("../data/stock_data.csv")
```

Relative paths are ambiguous inside containers. The working directory is unpredictable, and the `/data` volume may be mounted at a completely different location.

**Fix**: Replaced all hardcoded paths with environment variable lookups with a sensible default:

```python
import os
DATA_DIR = os.environ.get("DATA_DIR", "/data")
output_path = os.path.join(DATA_DIR, "stock_data.csv")
data.to_csv(output_path)
```

This way, the path can be overridden per environment without changing code.

---

#### Bug 2: yfinance MultiIndex Column Names

**Problem**: Modern versions of `yfinance` return a `pd.MultiIndex` for columns when downloading multiple tickers. The original code assumed flat column names like `"AAPL_Close"`, but the actual structure was `("Close", "AAPL")`.

```python
# Caused KeyError or wrong column names
data.columns  # -> MultiIndex([('Close', 'AAPL'), ('Open', 'AAPL'), ...])
```

**Fix**: Added a check and flattened the MultiIndex to single-level column names:

```python
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]
```

---

#### Bug 3: Positional Column Rename

**Problem**: The script renamed columns by position (e.g., `columns[0]` = "Open"), which broke when the column order changed across yfinance versions.

**Fix**: Used a dict-based rename that references columns by their actual name:

```python
data.rename(columns={"Adj Close": "AdjClose"}, inplace=True)
```

---

#### Bug 4: No Retry Logic

**Problem**: Network requests in containers are more fragile than on a laptop. A single transient failure would crash the whole pipeline step.

**Fix**: Wrapped the download call in a retry loop with delay:

```python
import time

MAX_RETRIES = 3
RETRY_DELAY = 15  # seconds

for attempt in range(MAX_RETRIES):
    try:
        data = fetch_data(ticker)
        break
    except Exception as e:
        if attempt < MAX_RETRIES - 1:
            print(f"Attempt {attempt+1} failed: {e}. Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
        else:
            raise
```

---

#### Bug 5: Missing User-Agent Header

**Problem**: yfinance requests sent from containers had no User-Agent header. Yahoo Finance's servers can detect and block these as bot traffic.

**Fix**: Created a custom session with a browser-like User-Agent:

```python
import requests

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
})
```

---

#### Final Fix: Switched from yfinance to Alpha Vantage API

**Problem**: Despite all the above fixes, Yahoo Finance continued to return empty responses (`YFTzMissingError: possibly delisted; no timezone found`). This was not a code bug — Yahoo Finance actively blocks IP address ranges associated with cloud providers and container environments.

**Root Cause**: Yahoo Finance detects datacenter/container IP ranges (including those used by k3d/Docker) as bot traffic and silently returns empty data.

**Fix**: Completely switched to the [Alpha Vantage API](https://www.alphavantage.co/), which provides free stock data:

```python
import requests
import pandas as pd
import os
import time

API_KEY = os.environ.get("ALPHA_VANTAGE_KEY")

def fetch_stock(ticker):
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY_ADJUSTED"
        f"&symbol={ticker}"
        f"&outputsize=compact"  # last 100 days (free tier)
        f"&apikey={API_KEY}"
    )
    response = requests.get(url)
    data = response.json()
    ts = data["Time Series (Daily)"]
    df = pd.DataFrame(ts).T
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

tickers = ["AAPL", "GOOGL", "MSFT", ...]
all_data = []

for ticker in tickers:
    df = fetch_stock(ticker)
    df.columns = [f"{ticker}_{col}" for col in df.columns]
    all_data.append(df)
    time.sleep(15)  # Alpha Vantage: 1 req/sec burst, 25 req/day on free tier

combined = pd.concat(all_data, axis=1)
combined.to_csv(os.path.join(DATA_DIR, "stock_data.csv"))
```

The API key was stored as a Kubernetes Secret and injected as an environment variable (see Part 4 and Part 5 for details).

---

### 3.2 `2_feature_engineering.py` — Feature Engineering Script

This script reads the combined stock CSV and computes technical indicators (RSI, MACD, Bollinger Bands, etc.) for each stock.

---

#### Bug 1: Incorrect Stock Detection with `"_Close" in col`

**Problem**: The script detected stock columns using:

```python
# Buggy
stock_cols = [col for col in df.columns if "_Close" in col]
```

The substring `"_Close"` also matched `"_AdjClose"` columns, causing the same stock to be processed twice and producing duplicate or incorrect features.

**Fix**: Changed to an exact suffix match that explicitly excludes `_AdjClose`:

```python
stock_cols = [
    col for col in df.columns
    if col.endswith("_Close") and not col.endswith("_AdjClose")
]
```

---

#### Bug 2: Unsafe Ticker Name Parsing

**Problem**: The script extracted ticker names from column names using positional splitting:

```python
ticker = col.split("_")[0]  # Breaks for tickers like "BRK_B"
```

This fails for tickers containing underscores.

**Fix**: Used `rsplit` to split from the right, keeping only the part before `_Close`:

```python
ticker = col.rsplit("_Close", 1)[0]
```

This correctly handles tickers like `BRK_B_Close` → `BRK_B`.

---

#### Bug 3: TA Library Type Incompatibility

**Problem**: The `ta` library (Technical Analysis library) requires float inputs. The data loaded from CSV had `object` dtype columns.

```python
# Caused: "unsupported operand type(s) for -: 'str' and 'str'"
ta.trend.macd(df["AAPL_Close"])
```

**Fix**: Explicitly cast columns to float before passing to TA functions:

```python
close = df[f"{ticker}_Close"].astype(float)
high = df[f"{ticker}_High"].astype(float)
low = df[f"{ticker}_Low"].astype(float)
```

---

### 3.3 `4_model_evaluation.py` — Model Evaluation Script

---

#### Bug 1: Filename Typo

**Problem**: The output file was named with a typo:

```python
output_path = os.path.join(DATA_DIR, "best_model_evalution.csv")  # typo
```

**Fix**: Corrected spelling:

```python
output_path = os.path.join(DATA_DIR, "best_model_evaluation.csv")
```

---

#### Bug 2: Series Saved Instead of DataFrame

**Problem**: The script saved evaluation results using single brackets, which returns a `pd.Series`:

```python
results_df.loc[best_idx].to_csv(output_path)  # saves Series (wrong shape)
```

A Series saved to CSV has the index as column names, not as a row, making it difficult to load back as a proper table.

**Fix**: Used double brackets to preserve DataFrame shape:

```python
results_df.loc[[best_idx]].to_csv(output_path, index=False)  # saves 1-row DataFrame
```

---

#### Bug 3: Hardcoded Paths

Same issue as in `1_fetch_data.py`. All paths replaced with `os.environ.get("DATA_DIR", "/data")`.

---

## 4. Part 2 — Production Services Built from Scratch

The original repository contained only the ML pipeline scripts. To make this a production-grade system, four additional services were built from scratch.

### 4.1 FastAPI Service (`services/fastapi/main.py`)

A REST API that serves model predictions and exposes Prometheus metrics.

**Prometheus Metrics Exposed:**

```python
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["endpoint"]
)
PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total predictions made",
    ["model", "stock"]
)
MODEL_MSE = Gauge("model_mse", "Model Mean Squared Error", ["model"])
MODEL_R2 = Gauge("model_r2", "Model R-squared score", ["model"])
```

**Endpoints:**

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check, returns `{"status": "ok"}` |
| `/metrics` | GET | Prometheus metrics (scraped by Prometheus server) |
| `/models` | GET | Lists all loaded models with performance scores |
| `/predict` | POST | Accepts `{stock, days}`, returns predicted prices |
| `/history/{stock}` | GET | Returns historical price data for a stock |

**Model Loading on Startup:**

```python
@app.on_event("startup")
async def load_models():
    models_dir = os.environ.get("MODELS_DIR", "/data/models")
    for model_file in os.listdir(models_dir):
        if model_file.endswith(".pkl"):
            model_name = model_file.replace(".pkl", "")
            with open(os.path.join(models_dir, model_file), "rb") as f:
                app.state.models[model_name] = pickle.load(f)
```

**Live Feature Computation:**

```python
def build_live_features(stock: str) -> pd.DataFrame:
    # Fetches 400 days of history to have enough data for all indicators
    # Computes RSI, MACD, Bollinger Bands, EMA, SMA
    # Returns the most recent row as feature vector for prediction
    ...
```

---

### 4.2 Streamlit Dashboard (`services/streamlit/app.py`)

A multi-tab web dashboard that visualizes predictions, model metrics, and pipeline status.

**Tabs:**

1. **Price History** — Plotly candlestick chart of historical stock prices
2. **Prediction** — Input form to select stock and prediction horizon; displays predicted prices as a line chart
3. **Model Metrics** — Bar charts comparing MSE and R² across all trained models
4. **Pipeline Status** — Shows Kubeflow pipeline run status and last run time

**Environment Variable:**

```python
FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://fastapi:8000")
```

All API calls go through this URL, making it easy to point to any FastAPI instance.

---

### 4.3 Prometheus Configuration (`services/prometheus/prometheus.yml`)

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "fastapi"
    static_configs:
      - targets: ["fastapi:8000"]
    metrics_path: "/metrics"
```

Prometheus scrapes the `/metrics` endpoint on the FastAPI service every 15 seconds.

---

### 4.4 Grafana Dashboard (`services/grafana/dashboards/stock_pipeline.json`)

A pre-built Grafana dashboard with 9 panels:

| Panel | Type | Description |
|---|---|---|
| Total Predictions | Stat | Cumulative prediction count |
| Request Rate | Graph | HTTP requests per second |
| Error Rate | Graph | 4xx/5xx responses per second |
| P95 Latency | Stat | 95th percentile response time |
| Requests by Endpoint | Bar Gauge | Per-endpoint request breakdown |
| Predictions by Model | Pie Chart | Which models are being used |
| Latency Percentiles | Graph | P50, P95, P99 latency over time |
| Model MSE | Bar Gauge | MSE for each trained model |
| Model R² | Bar Gauge | R² score for each trained model |

---

### 4.5 Docker Compose (`docker-compose.yml`)

All production services are orchestrated via Docker Compose for local development:

```yaml
version: "3.9"

services:
  fastapi:
    build: ./services/fastapi
    ports:
      - "8000:8000"
    environment:
      - MODELS_DIR=/data/models
      - DATA_DIR=/data
    volumes:
      - stock-data:/data

  streamlit:
    build: ./services/streamlit
    ports:
      - "8501:8501"
    environment:
      - FASTAPI_URL=http://fastapi:8000

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./services/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - ./services/grafana/dashboards:/var/lib/grafana/dashboards

volumes:
  stock-data:

# Pipeline components are under a separate profile
# docker-compose --profile pipeline up
```

---

## 5. Part 3 — Kubeflow Installation Nightmare

This was the most time-consuming part of the entire project — taking over 3 days of debugging. Installing Kubeflow Pipelines on k3d is not straightforward because many of Kubeflow's dependencies have subtle incompatibilities with containerized, resource-constrained environments.

### 5.1 Problem 1: busybox Init Container Blocked by `runAsNonRoot`

**Symptom**: The MySQL pod was stuck indefinitely in `Init:0/1` status:

```bash
kubectl get pods -n kubeflow
# NAME                     READY   STATUS       RESTARTS
# mysql-6d4f4b5c9-xk2p9    0/1     Init:0/1     0
```

**Cause**: The MySQL deployment included a busybox init container that ran as `root` to initialize the data directory. The pod security context had `runAsNonRoot: true`, which prevented root containers from running.

```bash
kubectl describe pod mysql-6d4f4b5c9-xk2p9 -n kubeflow
# Events:
#   Warning  Failed  CreateContainerConfigError
#   Error: container has runAsNonRoot and image will run as root
```

**Fix**: Patched the deployment to remove the init container entirely:

```bash
kubectl patch deployment mysql -n kubeflow \
  --type=json \
  -p='[{"op":"remove","path":"/spec/template/spec/initContainers"}]'
```

---

### 5.2 Problem 2: MySQL `auto.cnf` Self-Blocking Bug

**Symptom**: MySQL crashed immediately on every start, entering a `CrashLoopBackOff` state.

```bash
kubectl logs mysql-6d4f4b5c9-xk2p9 -n kubeflow
# [ERROR] Fatal error: Can't open and lock privilege tables: Table 'mysql.user' doesn't exist
# [ERROR] Aborting
```

**Cause**: This is a fundamental MySQL 8.0.x behavior. When MySQL starts, its very first action is to write `auto.cnf` (a file containing a UUID for server identification) to the data directory. After a container restart, MySQL finds `auto.cnf` already present and assumes the data directory is already initialized — so it skips the initialization step and tries to run normally. But the rest of the data directory is empty (because the PVC was freshly provisioned), causing MySQL to immediately crash.

This is not a bug unique to this project — it is a known issue with MySQL in Kubernetes environments where pods are restarted.

**Fix**: Used a two-phase init container approach:

1. The init container initializes MySQL into a tmpdir (`/tmp/mysqldata`) — a fresh directory with no `auto.cnf`.
2. After successful initialization, it copies everything to the real PVC mount.
3. The main container starts in "resume" mode (finds a fully initialized directory), skipping the empty-check.

```bash
# Init container command
mysqld --initialize-insecure \
  --datadir=/tmp/mysqldata \
  --innodb-use-native-aio=0

# Then copy to PVC
cp -a /tmp/mysqldata/. /var/lib/mysql/
```

---

### 5.3 Problem 3: AIO Failure in emptyDir Volumes

**Symptom**: The init container itself crashed:

```
InnoDB: Cannot initialize AIO sub-system
mysqld: Can't create/write to file '/tmp/mysqldata/ib_logfile0'
```

**Cause**: MySQL's InnoDB storage engine uses Linux's native Asynchronous I/O (AIO) subsystem for performance. However, Kubernetes `emptyDir` volumes are backed by the host's tmpfs or overlayfs, which do not support native AIO. This causes mysqld to fail immediately when it tries to initialize.

**Fix**: Disabled native AIO by adding `--innodb-use-native-aio=0` to both the init container command and the main container args. This forces MySQL to use simulated (synchronous) I/O, which is supported on all volume types:

```yaml
# In both init container command and main container args
- "--innodb-use-native-aio=0"
```

---

### 5.4 Problem 4: MySQL Version Incompatibilities

Multiple MySQL image versions were tested before finding one that worked. Each version had different authentication plugin support:

| Image | Problem |
|---|---|
| `mysql:8.4` | Completely removed `mysql_native_password` plugin — KFP cannot connect |
| `mysql:5.7` | Does not recognize `--authentication-policy` flag |
| `mysql:8.0.26` | `--authentication-policy` flag not supported (added in 8.0.27) |
| `mysql:8.0.36` | `--mysql-native-password=ON` is an unknown variable |
| `mysql:8.0.45` | Same unknown variable error as 8.0.36 |
| `mysql:8.0` (final) | Resolves to 8.0.45, works with `--authentication-policy=mysql_native_password` |

**Final working configuration:**

```yaml
image: mysql:8.0
args:
  - "--authentication-policy=mysql_native_password"
  - "--innodb-use-native-aio=0"
```

---

### 5.5 Problem 5: `mysql_native_password` Disabled in MySQL 8.0.34+

**Symptom**: After MySQL finally started, the `metadata-grpc-deployment` crashed:

```
rpc error: code = Unavailable
desc = mysql_native_password used by MLMD (ML Metadata) is not supported
as a default authentication plugin in this MySQL version
```

**Cause**: Starting from MySQL 8.0.34, `mysql_native_password` is disabled by default. Kubeflow's ML Metadata (MLMD) service — which is the backend database layer for tracking pipeline runs — requires `mysql_native_password` authentication to connect to MySQL.

**Tried**: Creating a ConfigMap with `[mysqld]\nmysql_native_password=ON` and mounting it as `/etc/mysql/conf.d/custom.cnf` — MySQL rejected this configuration with an "unknown variable" error.

**Fix**: Exec'd directly into the running MySQL pod and ran SQL commands to reconfigure user authentication:

```bash
kubectl exec -it mysql-6d4f4b5c9-xk2p9 -n kubeflow -- mysql -u root
```

```sql
-- Re-create root user with mysql_native_password plugin
ALTER USER 'root'@'localhost'
  IDENTIFIED WITH mysql_native_password BY '';

-- Create root user for remote connections (MLMD connects from another pod)
CREATE USER IF NOT EXISTS 'root'@'%'
  IDENTIFIED WITH mysql_native_password BY '';

GRANT ALL PRIVILEGES ON *.* TO 'root'@'%' WITH GRANT OPTION;
FLUSH PRIVILEGES;
```

---

### 5.6 Problem 6: SeaweedFS PVC Bound to Dead Node

**Symptom**: The `seaweedfs-master` pod was stuck in `Pending` state indefinitely:

```bash
kubectl describe pod seaweedfs-master-0 -n kubeflow
# Events:
#   Warning  FailedScheduling  0/2 nodes are available:
#   1 node(s) had volume node affinity conflict
```

**Cause**: SeaweedFS (the object storage backend used by Kubeflow for pipeline artifacts) uses a PersistentVolume with node affinity — meaning the PV is bound to a specific Kubernetes node. One of the k3d nodes (`b4503b2a36c6`) had gone `NotReady`. The PVC was still pinned to that dead node's affinity, so no other node could schedule the pod.

```bash
kubectl get nodes
# NAME              STATUS     ROLES
# k3d-stockcluster-server-0   Ready    control-plane
# b4503b2a36c6                NotReady  <none>       # dead node
```

**Fix**:

1. Delete the stuck PVC (required removing finalizers first, as Kubernetes holds PVCs until all pods release them):

```bash
kubectl patch pvc seaweedfs-master-pvc -n kubeflow \
  --type=json \
  -p='[{"op":"remove","path":"/metadata/finalizers"}]'

kubectl delete pvc seaweedfs-master-pvc -n kubeflow
```

2. Delete the orphaned PV:

```bash
kubectl delete pv <pv-name>
```

3. Create a fresh PVC without node affinity constraints, allowing Kubernetes to schedule on any available live node.

---

### 5.7 Problem 7: `ml-pipeline` OOM Kill (Exit Code 137)

**Symptom**: The main `ml-pipeline` pod (the Kubeflow Pipelines API server) died after approximately 90 seconds with no error logs. Exit code was 137.

```bash
kubectl get pods -n kubeflow
# NAME                        READY   STATUS      RESTARTS
# ml-pipeline-7d9c8b-lk4p2    0/1     OOMKilled   3
```

**Cause**: Exit code 137 = 128 + 9 = the process was killed by signal 9 (SIGKILL), which is what the Linux OOM (Out Of Memory) Killer sends. The default memory limit for the `ml-pipeline` container was `500Mi`, which was insufficient for the Kubeflow Pipelines API server to start up fully.

**Fix**: Patched the deployment to increase the memory limit to `2Gi`:

```bash
kubectl patch deployment ml-pipeline -n kubeflow \
  --type=json \
  -p='[{
    "op": "replace",
    "path": "/spec/template/spec/containers/0/resources/limits/memory",
    "value": "2Gi"
  }]'
```

---

### 5.8 Problem 8: `scheduledworkflow` and `persistenceagent` Timing Crashes

**Symptom**: Two supporting Kubeflow services — `scheduledworkflow` and `persistenceagent` — kept crashing with connection timeout errors.

**Cause**: Both services connect to `ml-pipeline` on startup to register themselves. Because `ml-pipeline` had been repeatedly restarting (due to OOM), these services tried to connect while `ml-pipeline` was unavailable and crashed. Once they were in `CrashLoopBackOff`, they continued crashing even after `ml-pipeline` stabilized, because Kubernetes exponential backoff kept them in a crash-restart cycle.

**Fix**: After `ml-pipeline` was confirmed stable, force-restarted both services:

```bash
kubectl rollout restart deployment scheduledworkflow -n kubeflow
kubectl rollout restart deployment persistenceagent -n kubeflow
```

This cleared their crash state and allowed them to connect cleanly to the now-stable `ml-pipeline`.

---

### 5.9 Final State

After all 8 problems were resolved:

```bash
kubectl get pods -n kubeflow
# NAME                                          READY   STATUS    RESTARTS
# cache-server-xxx                              1/1     Running   0
# metadata-envoy-deployment-xxx                 1/1     Running   0
# metadata-grpc-deployment-xxx                  1/1     Running   0
# metadata-writer-xxx                           1/1     Running   0
# ml-pipeline-xxx                               1/1     Running   0
# ml-pipeline-persistenceagent-xxx              1/1     Running   0
# ml-pipeline-scheduledworkflow-xxx             1/1     Running   0
# ml-pipeline-ui-xxx                            1/1     Running   0
# ml-pipeline-viewer-crd-xxx                    1/1     Running   0
# ml-pipeline-visualizationserver-xxx           1/1     Running   0
# minio-xxx                                     1/1     Running   0
# mysql-xxx                                     1/1     Running   0
# workflow-controller-xxx                       1/1     Running   0
# seaweedfs-master-xxx                          1/1     Running   0
# Total: 14 pods, all Running
```

Kubeflow UI accessible at: `http://localhost:8080`

Port-forwarding command:

```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

---

## 6. Part 4 — Building the KFP Pipeline

With Kubeflow running, the next step was to define the ML pipeline using the KFP SDK v2 and deploy it to the cluster.

### 6.1 Pipeline Definition (`pipeline/pipeline.py`)

The pipeline was defined as a directed acyclic graph (DAG) with the following steps:

```
fetch_data
    |
feature_engineering
    |
    +---> train_xgboost
    +---> train_gradient_boosting    --> model_evaluation
    +---> train_random_forest
```

**Key design decisions:**

- All steps share a single PVC `stock-pipeline-data` mounted at `/data`
- Images are loaded into k3d via `k3d image import`, so `imagePullPolicy: IfNotPresent` is used
- Training steps run in parallel (no dependency between them, only both depend on feature engineering)

**Pipeline skeleton:**

```python
from kfp import dsl
from kfp.kubernetes import use_pvc_as_volume, set_image_pull_policy, CreatePVC

@dsl.pipeline(name="stock-market-pipeline")
def stock_pipeline():
    pvc = CreatePVC(
        pvc_name="stock-pipeline-data",
        access_modes=["ReadWriteOnce"],
        storage="5Gi"
    )

    fetch = fetch_data_component()
    use_pvc_as_volume(fetch, pvc_name="stock-pipeline-data", mount_path="/data")
    set_image_pull_policy(fetch, "IfNotPresent")

    features = feature_engineering_component()
    features.after(fetch)
    use_pvc_as_volume(features, pvc_name="stock-pipeline-data", mount_path="/data")

    xgb = train_xgboost_component()
    xgb.after(features)
    use_pvc_as_volume(xgb, pvc_name="stock-pipeline-data", mount_path="/data")

    gb = train_gradient_boosting_component()
    gb.after(features)
    use_pvc_as_volume(gb, pvc_name="stock-pipeline-data", mount_path="/data")

    rf = train_random_forest_component()
    rf.after(features)
    use_pvc_as_volume(rf, pvc_name="stock-pipeline-data", mount_path="/data")

    evaluation = model_evaluation_component()
    evaluation.after(xgb, gb, rf)
    use_pvc_as_volume(evaluation, pvc_name="stock-pipeline-data", mount_path="/data")
```

---

### 6.2 Errors Encountered While Building the Pipeline

#### Error 1: `kfp_kubernetes` ModuleNotFoundError

**Problem**:

```
ModuleNotFoundError: No module named 'kfp_kubernetes'
```

**Cause**: The package `kfp-kubernetes` installs its Python module under the path `kfp/kubernetes/`, not as a top-level `kfp_kubernetes` module. This is non-standard and caused confusion.

**Fix**: Changed the import statement:

```python
# Wrong
from kfp_kubernetes import use_pvc_as_volume

# Correct
from kfp.kubernetes import use_pvc_as_volume
```

---

#### Error 2: `dsl.EnvVariable` AttributeError

**Problem**:

```
AttributeError: module 'kfp.dsl' has no attribute 'EnvVariable'
```

**Cause**: `dsl.EnvVariable` was removed in KFP SDK v2. It was available in v1 only.

**Fix**: Removed the use of `dsl.EnvVariable` entirely. Instead, all pipeline scripts use `os.environ.get("VAR_NAME", "default")`, which reads environment variables that can be injected at the container level through other mechanisms.

---

#### Error 3: `CreatePVC(name=...)` TypeError

**Problem**:

```
TypeError: CreatePVC() got an unexpected keyword argument 'name'
```

**Cause**: In the KFP SDK v2 `CreatePVC` component, the parameter for specifying the PVC name is `pvc_name`, not `name`. The `name` parameter in KFP refers to the component task name, not the PVC resource name.

**Fix**:

```python
# Wrong
pvc = CreatePVC(name="stock-pipeline-data", ...)

# Correct
pvc = CreatePVC(pvc_name="stock-pipeline-data", ...)
```

---

#### Error 4: UnicodeEncodeError on Arrow Character

**Problem**:

```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position 47
```

**Cause**: The pipeline script used `→` (Unicode arrow, U+2192) in a comment. Windows uses CP1252 as the default code page, which cannot encode this character. The error occurred when Python tried to print or log the comment.

**Fix**: Replaced all Unicode arrows with ASCII:

```python
# Wrong (in comments or print statements)
# fetch_data → feature_engineering

# Correct
# fetch_data > feature_engineering
```

---

#### Error 5: `set_image_pull_policy` Wrong Module

**Problem**:

```
ImportError: cannot import name 'set_image_pull_policy' from 'kfp.dsl'
```

**Cause**: `set_image_pull_policy` is a Kubernetes-specific extension, not part of the core KFP DSL.

**Fix**: Corrected the import:

```python
# Wrong
from kfp.dsl import set_image_pull_policy

# Correct
from kfp.kubernetes import set_image_pull_policy
```

---

#### Error 6: 409 Conflict on Pipeline Version Upload

**Problem**:

```
ApiException: (409) Conflict — pipeline version with the name 'latest' already exists
```

**Cause**: On first upload, the pipeline was given version name `"latest"`. On subsequent re-uploads (to fix bugs), the version name `"latest"` already existed in Kubeflow and the API rejected the duplicate.

**Fix**: Used timestamp-based version names to ensure uniqueness on every upload:

```python
from datetime import datetime

version_name = datetime.now().strftime("v%Y%m%d-%H%M%S")
# e.g., "v20260401-143022"

client.upload_pipeline_version(
    pipeline_package_path="pipeline.yaml",
    pipeline_id=pipeline_id,
    pipeline_version_name=version_name
)
```

---

#### Error 7: PowerShell `&&` Not Supported

**Problem**:

```
The token '&&' is not a valid statement separator in this version of PowerShell.
```

**Cause**: The `&&` operator for command chaining is a bash/sh convention. Windows PowerShell does not support `&&`.

**Fix**: Used semicolon (`;`) in PowerShell, or switched to Git Bash / WSL for running pipeline commands:

```powershell
# Wrong in PowerShell
python compile.py && python upload.py

# Correct in PowerShell
python compile.py; python upload.py

# Or use Git Bash / WSL
python compile.py && python upload.py
```

---

#### Error 8: `use_secret_as_env` Wrong Kwargs

**Problem**:

```
TypeError: use_secret_as_env() got unexpected keyword arguments
```

**Cause**: The function signature was being called with incorrect argument names.

**Fix**: Used the correct signature:

```python
from kfp.kubernetes import use_secret_as_env

use_secret_as_env(
    task=fetch_task,
    secret_name="alpha-vantage",
    secret_key_to_env={"api-key": "ALPHA_VANTAGE_KEY"}
)
```

This maps the Kubernetes Secret key `api-key` from the Secret named `alpha-vantage` to the environment variable `ALPHA_VANTAGE_KEY` inside the container.

---

## 7. Part 5 — fetch-data Internet Access Issues

Even after the pipeline was running, the `fetch-data` step failed to download any stock data. This introduced a new category of problem: container networking and external API access.

### 7.1 Problem 1: DNS Resolution Failure in Pods

**Symptom**: The fetch-data pod failed with:

```
socket.gaierror: [Errno -2] Name or service not known
requests.exceptions.ConnectionError: HTTPSConnectionPool(host='fc.yahoo.com', port=443):
  Max retries exceeded -- No address associated with hostname
```

**Cause**: CoreDNS — the Kubernetes internal DNS server — was configured to forward external DNS queries to the address in `/etc/resolv.conf` of the host node. In a k3d environment, the host's `/etc/resolv.conf` pointed to Docker's internal DNS resolver (`127.0.0.11`), which is only valid inside Docker containers, not inside Kubernetes pods. The pods could not resolve any external hostnames.

**Diagnosis:**

```bash
kubectl exec -it <fetch-pod> -n kubeflow -- nslookup google.com
# ;; connection timed out; no servers could be reached
```

**Fix**: Patched the CoreDNS ConfigMap to use Google's public DNS servers directly:

```bash
kubectl edit configmap coredns -n kube-system
```

Changed the `forward` line in the Corefile from:

```
forward . /etc/resolv.conf
```

To:

```
forward . 8.8.8.8 8.8.4.4
```

Then restarted CoreDNS:

```bash
kubectl rollout restart deployment coredns -n kube-system
```

After this fix, pods could resolve external hostnames.

---

### 7.2 Problem 2: Yahoo Finance Blocks Container IPs

**Symptom**: Even after DNS was fixed, yfinance returned empty data:

```
YFTzMissingError: $AAPL: possibly delisted; no timezone found $AAPL
```

No actual error was raised — yfinance simply returned an empty DataFrame as if the ticker didn't exist.

**Cause**: Yahoo Finance actively detects and blocks IP address ranges associated with cloud providers, container environments, and datacenters. k3d pods run on Docker's internal network, which uses IP ranges known to Yahoo Finance as non-user (bot) traffic. They silently return empty responses rather than raising an HTTP error.

**Attempted Fix 1**: User-Agent header spoofing (already described in Part 1) — did not work. Yahoo's blocking is IP-based, not User-Agent-based.

**Attempted Fix 2**: `curl_cffi` browser impersonation:

```python
# Tried to use curl_cffi to impersonate a real browser's TLS fingerprint
import curl_cffi.requests as requests
session = requests.Session(impersonate="chrome110")
```

This failed with:

```
AttributeError: 'str' object has no attribute 'name'
```

**Root Cause of curl_cffi failure**: The `yfinance` library internally calls `session.headers.name` (treating it as an object), but `curl_cffi`'s session object is not fully compatible with `requests.Session`. The interface mismatch caused the crash.

**Root cause of Yahoo blocking**: This is a known, documented, and intentional behavior by Yahoo Finance. There is no reliable workaround for cloud/container IPs.

---

### 7.3 Problem 3: Alpha Vantage Free Tier Limitations

After switching to Alpha Vantage (see Part 1), two additional constraints were encountered:

**Limitation 1**: `outputsize=full` (up to 20 years of data) is a premium-only feature. The free tier only allows `outputsize=compact` (last 100 trading days, approximately 100 rows).

**Fix**: Changed to `outputsize=compact`. 100 rows is sufficient for feature engineering and model training in this project.

**Limitation 2**: Free tier rate limit is 25 requests per day and approximately 1 request per second burst. With 10 stocks, 10 API calls are needed per pipeline run.

**Fix**: Added a 15-second delay between requests to stay well within limits:

```python
for ticker in tickers:
    data = fetch_stock(ticker)
    all_data.append(data)
    time.sleep(15)  # Respect rate limit
```

---

### 7.4 Storing the API Key as a Kubernetes Secret

API keys should never be hardcoded in pipeline code or Docker images. The Alpha Vantage API key was stored as a Kubernetes Secret:

```bash
kubectl create secret generic alpha-vantage \
  --from-literal=api-key=YOUR_API_KEY_HERE \
  -n kubeflow
```

The secret was then injected into the fetch-data pipeline step as an environment variable using `use_secret_as_env`:

```python
from kfp.kubernetes import use_secret_as_env

fetch_task = fetch_data_component()
use_secret_as_env(
    task=fetch_task,
    secret_name="alpha-vantage",
    secret_key_to_env={"api-key": "ALPHA_VANTAGE_KEY"}
)
```

Inside the container, the script reads:

```python
API_KEY = os.environ.get("ALPHA_VANTAGE_KEY")
```

---

### 7.5 Final Result

After all fixes in Part 5:

- DNS resolution works in all pods
- Alpha Vantage API successfully responds from container IPs
- All 10 stocks downloaded (100 rows each)
- Combined CSV with all stock data saved to PVC at `/data/stock_data.csv`
- `fetch-data` pipeline step: **SUCCEEDED**

---

## 8. Part 5b — Docker Image Patching Without Internet

After switching to Alpha Vantage and fixing the feature engineering script, the Docker images for each pipeline component still had the old code inside them. Normally, you would rebuild them with `docker build`. But Docker build pulls the base image (`python:3.11-slim`) from Docker Hub — and Docker Hub was unreachable from this network environment.

### 8.1 The Problem

```
docker build -t stock/fetch-data:latest ./components/1_fetch_data
# Step 1/8 : FROM python:3.11-slim
# ERROR: failed to solve: python:3.11-slim: unexpected status code 403 Forbidden
```

Every `docker build` command failed at the very first step (`FROM python:3.11-slim`) because Docker Hub was blocked. This meant we could not rebuild any image from scratch.

### 8.2 The Solution: `docker commit` Patching

Instead of rebuilding from a Dockerfile, we patched the existing running images by:

1. **Running the old image** as a temporary container
2. **Copying the updated source file** into the running container with `docker cp`
3. **Committing the container** to a new image tag with `docker commit`
4. **Importing the new image into k3d** with `k3d image import`

This approach works because the base image and all pip-installed packages are already inside the existing container — only the source `.py` file changed.

**Step-by-step for `feature_engineering` image:**

```powershell
# 1. Start the old image as a detached container (don't run the script, just sleep)
docker run -d --name patch-fe --entrypoint sleep stock/feature-engineering:latest 3600

# 2. Copy the updated Python file into the running container
docker cp components/2_feature_engineering/src/feature_engineering.py patch-fe:/app/src/feature_engineering.py

# 3. Commit the container to a new image
docker commit patch-fe stock/feature-engineering:latest

# 4. Remove the temp container
docker rm -f patch-fe

# 5. Import the patched image into k3d so Kubeflow pods can use it
k3d image import stock/feature-engineering:latest -c stockcluster
```

**Step-by-step for `train_gradient_boosting` image** (same pattern):

```powershell
docker run -d --name patch-gb --entrypoint sleep stock/train-gradient-boosting:latest 3600
docker cp components/3_train_gradient_boosting/src/train_gradient_boosting.py patch-gb:/app/src/train_gradient_boosting.py
docker commit patch-gb stock/train-gradient-boosting:latest
docker rm -f patch-gb
k3d image import stock/train-gradient-boosting:latest -c stockcluster
```

### 8.3 Why This Works

When you run `docker commit`, Docker snapshots the container's current filesystem layer — including any files you copied in with `docker cp`. The resulting image is identical to the original except for those changed files. Since all Python packages were already installed in the original image (via `pip install -r requirements.txt`), the patched image works immediately without needing internet access.

### 8.4 Important Note for Production

In production, `docker commit` is considered an anti-pattern because it creates images that are not reproducible from a Dockerfile. The correct solution is to fix the network access (or use a private registry mirror) so `docker build` can run normally. In this project, `docker commit` was used as a workaround specifically because Docker Hub was blocked in the development environment.

---

## 9. Part 5c — NaN Training Failures

After fixing the fetch-data step (Alpha Vantage API) and feature engineering step (column detection), the pipeline progressed to the training steps. Three training steps ran in parallel. Two succeeded (XGBoost and Random Forest) but **Gradient Boosting failed**.

### 9.1 The Gradient Boosting NaN Crash

**Symptom** (from pod logs):

```
ValueError: Input X contains NaN.
GradientBoostingRegressor does not accept missing values encoded as NaN natively.
```

**Initial diagnosis**: The feature engineering output had NaN values in some rows (from the TA indicator warmup period). XGBoost and Random Forest handle NaN internally — Gradient Boosting does not.

**Initial fix attempt**: Added a NaN mask to the gradient boosting training script:

```python
mask = X.notna().all(axis=1) & y.notna()
X, y = X[mask], y[mask]
print(f"Dataset shape after dropping NaN: {X.shape}")
```

**Result of the fix**:

```
Dataset shape after dropping NaN: (0, 109)
```

Zero rows survived the NaN filter. This meant **every single row** in the dataset had at least one NaN column — the mask dropped everything.

### 9.2 Root Cause Investigation

The question became: why does every row have a NaN?

**Step 1: Understand the data size.**

Alpha Vantage free tier returns `outputsize=compact` = **100 rows** per stock (last 100 trading days). This is the starting point.

**Step 2: Understand the TA indicator warmup.**

Technical Analysis indicators need historical data to "warm up" before producing valid values:

| Indicator | Window | Rows needed before valid output |
|---|---|---|
| MA20 | 20 | First 19 rows = NaN |
| MA50 | 50 | First 49 rows = NaN |
| RSI(14) | 14 | First 13 rows = NaN |
| MACD | slow=26 | First 25 rows = NaN |

With 100 rows and MA50, after warmup you have at most `100 - 49 = 51` valid rows per stock.

**Step 3: Understand the multi-stock outer join.**

The 10 stocks are merged with `how="outer"` on Date. This is correct — different stocks have different trading days (US stocks don't trade on Indian market holidays and vice versa). But outer join introduces NaN everywhere a stock has no data for a given date.

**Step 4: The combined effect.**

After the outer join, each stock's warmup NaNs now appear in the combined DataFrame across all columns for those dates. The original code used:

```python
df = df.iloc[50:]  # remove first 50 rows
```

But `iloc[50:]` removes the first 50 *DataFrame* rows — it does not guarantee that the TA indicator warmup NaNs are gone, because after the outer join, NaN positions can appear anywhere in the date range, not just at the beginning.

**Final result**: Every row had at least one NaN somewhere across the 109 feature columns.

### 9.3 The Fix

**Two-part fix:**

**Fix 1 — Reduce MA windows** in `feature_engineering.py`:

```python
# Before (too large for 100-row dataset)
df[f"{stock}_MA50"] = close.rolling(window=50).mean()
df[f"{stock}_MA20"] = close.rolling(window=20).mean()

# After (fits within 100 rows with buffer to spare)
df[f"{stock}_MA10"] = close.rolling(window=10).mean()
df[f"{stock}_MA20"] = close.rolling(window=20).mean()
```

MA10 only needs 9 warmup rows. Combined with MACD's 25-row warmup, we need ~26 rows minimum, leaving ~74 valid rows from 100.

**Fix 2 — Replace `iloc[50:]` with `ffill().bfill().dropna()`**:

```python
# Before (removes rows by position, doesn't guarantee NaN-free)
df = df.iloc[50:].reset_index(drop=True)

# After (forward-fill, backward-fill for outer-join gaps, then drop any remaining NaN)
df = df.ffill().bfill().dropna().reset_index(drop=True)
```

`ffill()` fills gaps from the previous valid row (handles dates where one stock had no trading). `bfill()` fills any remaining NaN at the start. `dropna()` removes any rows that still have NaN after both fills.

**After the fix:**

```
Final shape: (74, 109)
Dataset shape after dropping NaN: (74, 109)  # 74 valid rows, 109 features
```

All three training steps completed:
```
XGBoost  MSE: 5.4309  RMSE: 2.3305
Random Forest  MSE: 5.4309  RMSE: 2.3305
Gradient Boosting  MSE: 3.3784  RMSE: 1.8382
```

### 9.4 The run_pipeline.ps1 Port-Forward Detection Bug

While rerunning the pipeline after fixes, the automation script `run_pipeline.ps1` failed to detect whether Kubeflow's port-forward was already running.

**Problem**: The script checked for existing port-forward processes using:

```powershell
$existing = Get-Process | Where-Object { $_.CommandLine -like "*port-forward*" }
```

**Error**:
```
InvalidOperation: You cannot call a method on a null-valued expression.
```

**Cause**: On Windows PowerShell, `Get-Process` objects do not have a `CommandLine` property by default — `$_.CommandLine` is always `$null`. This is a known Windows PowerShell limitation; `CommandLine` is available on `Get-WmiObject Win32_Process`, not on `Get-Process`.

**Fix**: Replaced the process detection with a simple HTTP reachability test:

```powershell
# Before (broken on Windows PowerShell)
$existing = Get-Process | Where-Object { $_.CommandLine -like "*port-forward*" }

# After (works on any OS - just test if the port is reachable)
try {
    Invoke-WebRequest -Uri "http://localhost:8080" -TimeoutSec 2 -UseBasicParsing | Out-Null
    Write-Host "Port-forward already active."
} catch {
    # Not reachable - start port-forward
    Start-Process kubectl -ArgumentList "port-forward -n kubeflow svc/ml-pipeline-ui 8080:80" -WindowStyle Hidden
    Start-Sleep 3
}
```

### 9.5 Final Pipeline Run — All Steps Green

After applying all fixes (Alpha Vantage API, MA10/MA20 feature engineering, ffill/bfill NaN handling, docker commit image patching), the full 6-step pipeline ran to completion:

```
Step 1: fetch-data              SUCCESS  (Alpha Vantage, 10 stocks, 100 rows each)
Step 2: feature-engineering     SUCCESS  (74 valid rows after NaN handling)
Step 3a: train-xgboost          SUCCESS  (MSE: 5.43, RMSE: 2.33, R2: 0.944)
Step 3b: train-gradient-boost   SUCCESS  (MSE: 3.38, RMSE: 1.84, R2: 0.965)
Step 3c: train-random-forest    SUCCESS  (MSE: 5.43, RMSE: 2.33, R2: 0.944)
Step 4: model-evaluation        SUCCESS  (Best: Gradient Boosting, saved results)
```

All 6 nodes showing green in the Kubeflow UI. Gradient Boosting was selected as the best model with R² = 0.965.

---

## 10. Key Concepts Learned

This project involved many Kubernetes, MySQL, and MLOps concepts that are not commonly covered in standard tutorials. This section summarizes the most important ones.

---

### PVC (PersistentVolumeClaim)

A PVC is a Kubernetes object that reserves a piece of persistent storage. Think of it as a "reserved disk" that survives pod restarts. In this pipeline, all steps share a single PVC, allowing data to flow from fetch → features → training → evaluation without needing a separate object store (like MinIO or S3) for intermediate results.

```bash
# Create a PVC
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: stock-pipeline-data
  namespace: kubeflow
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 5Gi
EOF
```

---

### auto.cnf — MySQL's Self-Blocking UUID File

`auto.cnf` is a file MySQL writes to its data directory on very first start. It contains a unique server UUID. The problem: MySQL writes this file before checking if the data directory is properly initialized. On a pod restart with a fresh/empty PVC, MySQL finds `auto.cnf` (if it somehow persists) and assumes initialization is done, then crashes when it finds nothing else. The fix was to initialize into a tmpdir first, then copy — ensuring the PVC always has a complete, consistent MySQL data directory.

---

### AIO (Asynchronous I/O)

Linux native AIO is a kernel feature that allows applications to submit I/O operations and continue without waiting for them to complete. MySQL's InnoDB engine uses this for performance. However, many Kubernetes volume types (especially `emptyDir` backed by tmpfs or overlayfs) do not support native AIO. This causes MySQL to fail at startup. Solution: `--innodb-use-native-aio=0`.

---

### `runAsNonRoot`

A Kubernetes pod security policy that prevents containers from running as the root user (UID 0). This is a security best practice but can break init containers that legitimately need root (like busybox filesystem init containers). Solution: remove the init container or redesign it to run as a non-root user.

---

### `mysql_native_password`

The original (legacy) MySQL authentication plugin. Kubeflow's ML Metadata (MLMD) service was built to use this plugin and does not support MySQL's newer `caching_sha2_password` (the default in MySQL 8.0.4+) or `auth_iam`. This incompatibility was the root cause of several crashes. The solution was to either use an older MySQL version or manually reconfigure user authentication after MySQL started.

---

### Node Affinity

A Kubernetes scheduling constraint that binds a PersistentVolume to a specific node. When that node dies (or becomes `NotReady`), any pod requiring that PV can never be scheduled — because the PV is physically on the dead node and no other node can serve it. Solution: delete the PVC/PV and create fresh ones without node affinity, or use a storage class that supports dynamic provisioning across nodes.

---

### OOM Kill (Exit Code 137)

Exit code 137 = 128 + signal 9 (SIGKILL). This means the Linux kernel's Out-Of-Memory killer terminated the process. In Kubernetes, this happens when a container exceeds its memory `limit`. The container is killed immediately with no warning or graceful shutdown. Solution: increase the memory limit in the deployment spec.

---

### CoreDNS

Kubernetes' built-in DNS server that resolves service names (`ml-pipeline.kubeflow.svc.cluster.local`) and forwards external queries to upstream DNS servers. In k3d, the default upstream (from the host's `/etc/resolv.conf`) is Docker's internal DNS, which is not accessible to pods. Fix: configure CoreDNS to forward to public DNS (8.8.8.8).

---

### k3d

k3d is a tool that runs k3s (a lightweight Kubernetes distribution) inside Docker containers. This enables running a full Kubernetes cluster on a development laptop without a separate VM or cloud provider. k3d is fast to start and lightweight, but has limitations including no support for native AIO in volumes and restricted network access.

---

### KFP SDK v2

The Python SDK for defining Kubeflow Pipelines as Python code. Key changes from v1 to v2:

- Components are defined with `@dsl.component` decorator
- Kubernetes-specific features (PVCs, secrets, image pull policy) are in `kfp.kubernetes`, not `kfp.dsl`
- `dsl.EnvVariable` was removed
- `CreatePVC` uses `pvc_name` as the static name input parameter

---

## 11. Summary Table of All Problems and Fixes

| # | Component | Problem | Root Cause | Fix |
|---|---|---|---|---|
| 1 | fetch_data | Hardcoded `../data/` paths | Relative paths invalid in containers | `os.environ.get("DATA_DIR", "/data")` |
| 2 | fetch_data | MultiIndex column names from yfinance | Modern yfinance returns MultiIndex for multi-ticker download | Flatten with `[col[0] for col in data.columns]` |
| 3 | fetch_data | Positional column rename breaks | Column order not guaranteed | Dict-based rename by column name |
| 4 | fetch_data | No retry on network failure | Single transient failure crashes the step | 3-attempt retry loop with 15s delay |
| 5 | fetch_data | Bot detection by Yahoo Finance | No User-Agent header | Added browser-like User-Agent to session |
| 6 | fetch_data | Yahoo Finance blocks container IPs | Yahoo actively blocks cloud/container IPs | Switched entirely to Alpha Vantage API |
| 7 | feature_engineering | `"_Close" in col` matches `_AdjClose` | Substring match too broad | `col.endswith("_Close") and not col.endswith("_AdjClose")` |
| 8 | feature_engineering | Wrong ticker name for tickers with `_` | `split("_")[0]` splits on first underscore | `rsplit("_Close", 1)[0]` splits from the right |
| 9 | feature_engineering | TA library crashes on string columns | CSV loads all columns as `object` dtype | `.astype(float)` before passing to TA functions |
| 10 | model_evaluation | Output file not found downstream | Filename typo: `evalution` vs `evaluation` | Corrected spelling |
| 11 | model_evaluation | Wrong CSV shape for best model | `loc[idx]` returns Series, not DataFrame | `loc[[idx]]` (double brackets) returns DataFrame |
| 12 | model_evaluation | Hardcoded paths | Same as fetch_data | `os.environ.get("DATA_DIR", "/data")` |
| 13 | Kubeflow Install | MySQL pod stuck in `Init:0/1` | busybox init container runs as root; blocked by `runAsNonRoot` | Remove init container via `kubectl patch` |
| 14 | Kubeflow Install | MySQL crash loop after restart | `auto.cnf` tricks MySQL into skipping initialization | Init to tmpdir, copy to PVC, main container sees complete dir |
| 15 | Kubeflow Install | MySQL fails to start in emptyDir | emptyDir volumes don't support native Linux AIO | `--innodb-use-native-aio=0` |
| 16 | Kubeflow Install | Various MySQL version errors | Different versions have incompatible auth flags | Final: `mysql:8.0` with `--authentication-policy=mysql_native_password` |
| 17 | Kubeflow Install | `metadata-grpc` crashes | MySQL 8.0.34+ disables `mysql_native_password` by default | Exec into MySQL, run `ALTER USER` and `CREATE USER` SQL |
| 18 | Kubeflow Install | seaweedfs stuck in Pending | PVC had node affinity to dead node | Delete PVC/PV (remove finalizers), create fresh PVC |
| 19 | Kubeflow Install | `ml-pipeline` OOMKilled (exit 137) | Default memory limit 500Mi too small | Patch memory limit to `2Gi` |
| 20 | Kubeflow Install | `scheduledworkflow`/`persistenceagent` crash | Tried connecting to ml-pipeline before it was ready | `kubectl rollout restart` after ml-pipeline stabilized |
| 21 | Pipeline SDK | `ModuleNotFoundError: kfp_kubernetes` | Package installs under `kfp/kubernetes/` not `kfp_kubernetes/` | `from kfp.kubernetes import ...` |
| 22 | Pipeline SDK | `dsl.EnvVariable` AttributeError | Removed in KFP SDK v2 | Remove usage; use `os.environ.get` in scripts |
| 23 | Pipeline SDK | `CreatePVC(name=...)` TypeError | Input param is `pvc_name`, not `name` | `CreatePVC(pvc_name="...", ...)` |
| 24 | Pipeline SDK | UnicodeEncodeError on `→` | Windows CP1252 can't encode Unicode arrow | Replace with ASCII `>` |
| 25 | Pipeline SDK | `set_image_pull_policy` ImportError | Function is in `kfp.kubernetes`, not `kfp.dsl` | `from kfp.kubernetes import set_image_pull_policy` |
| 26 | Pipeline SDK | 409 Conflict on version upload | Version name "latest" already exists | Timestamp-based version names `v%Y%m%d-%H%M%S` |
| 27 | Pipeline SDK | `&&` not supported | PowerShell uses `;` not `&&` | Use `;` in PowerShell or switch to Git Bash |
| 28 | Pipeline SDK | `use_secret_as_env` TypeError | Wrong keyword argument names | `use_secret_as_env(task, secret_name=..., secret_key_to_env={...})` |
| 29 | Pod Networking | DNS resolution fails in pods | CoreDNS forwarding to Docker-internal DNS unreachable from pods | Patch CoreDNS ConfigMap to use `8.8.8.8 8.8.4.4` |
| 30 | Pod Networking | Yahoo Finance returns empty data | Yahoo blocks datacenter/container IPs as bots | Switch to Alpha Vantage API |
| 31 | Pod Networking | `curl_cffi` session incompatibility | `curl_cffi` session interface incompatible with yfinance internals | Abandoned approach; use Alpha Vantage directly |
| 32 | Alpha Vantage | `outputsize=full` requires premium | Free tier only supports compact (100 days) | Use `outputsize=compact` |
| 33 | Alpha Vantage | Rate limit (25 req/day) | Free tier restriction | Add `time.sleep(15)` between stock requests |
| 34 | Docker Build | `docker build` fails — cannot pull python:3.11-slim | Docker Hub blocked from this network (403 Forbidden on base image pull) | `docker run` existing image + `docker cp` updated .py + `docker commit` to create patched image tag |
| 35 | run_pipeline.ps1 | `$_.CommandLine` is null — cannot detect port-forward | `Get-Process` on Windows doesn't expose CommandLine property | Replaced process scan with `Invoke-WebRequest -Uri http://localhost:8080` reachability test |
| 36 | feature_engineering | All rows NaN after MA50 warmup | 100 rows of data not enough for 50-row window; multi-stock merge multiplies gaps | Changed MA50/MA20 to MA20/MA10; replaced `iloc[50:]` with `ffill().bfill().dropna()` |
| 35 | train_gradient_boosting | GradientBoostingRegressor crashes on NaN | Sklearn GB does not handle NaN unlike XGBoost | Added NaN mask; root fix was in feature_engineering (row 34 above) |
| 36 | docker build | Cannot pull python:3.11-slim base image | Docker Hub unreachable from this network | Used `docker run` existing image + `docker cp` updated files + `docker commit` |
| 37 | kubectl cp | "one of src or dest must be local" on Windows | Drive letter `d:` looks like a remote host:path spec to kubectl | Used relative path `./data/models` from project root instead of absolute path |
| 38 | kubectl run --overrides | "Invalid JSON Patch" on PowerShell | PowerShell strips/mangles single-quoted JSON in `--overrides` arg | Created YAML pod manifest file; used `kubectl apply -f pod.yaml` instead |
| 39 | Docker Compose | Grafana fails to load app files | Port remapped to 3001 but Grafana still internally uses root_url with port 3000 | Added `GF_SERVER_ROOT_URL: http://localhost:3001` env var to Grafana service |
| 40 | Docker Compose | FastAPI healthcheck fails (container unhealthy) | `curl` not installed in `python:3.11-slim` image | Changed healthcheck to use `python -c "import urllib.request; urllib.request.urlopen(...)"` |
| 41 | Docker Compose | Port 3000 already in use | Another service (VS Code Live Share or similar) bound to 3000 | Remapped Grafana to `3001:3000` in docker-compose.yml |
| 42 | FastAPI /history | 404 "No data found for AAPL" | FastAPI used yfinance for live data; yfinance blocked from Docker containers same as in Kubeflow | Replaced yfinance call with local CSV read from `/data/combined/all_10_stocks_10years.csv` |
| 43 | FastAPI /predict | Feature mismatch on predict | FastAPI computed MA50/MA20 but models trained on MA10/MA20 | Fixed FastAPI feature engineering to use MA10/MA20 matching training pipeline |
| 44 | kubectl cp combined/features | `wsarecv` connection forcibly closed | Running two `kubectl cp` commands back-to-back overwhelms the k3d API server tunnel | Run each `kubectl cp` separately, not chained with `>>` |

---

## 12. Part 6 — Connecting the Serving Layer

After the Kubeflow pipeline completed successfully with all 6 steps green, the serving layer (FastAPI, Streamlit, Prometheus, Grafana) was connected to the trained models.

### 12.1 Architecture

```
[Kubeflow PVC: stock-data]
        |
        | kubectl cp (one-time export)
        v
[d:\Stock_Kubeflow\data\]
    models/
        XGBoost_model.pkl
        random_forest_model.pkl
        gradient_boosting_model.pkl
        all_models_results.csv
        best_model.csv
    combined/
        all_10_stocks_10years.csv
    features/
        stock_features.csv
        |
        | bind-mounted as /data
        v
[Docker Compose]
    FastAPI  :8000  -- loads models, serves predictions from local data
    Streamlit :8501  -- UI dashboard, calls FastAPI
    Prometheus :9090 -- scrapes /metrics from FastAPI every 15s
    Grafana  :3001  -- dashboards showing model MSE, R2, request rates
```

The Kubeflow pipeline runs on Kubernetes (k3d). The serving layer runs on Docker Compose on the host machine. They share data via a local directory that was exported from the Kubernetes PVC using `kubectl cp`.

---

### 12.2 Step 1: Export Models from Kubernetes PVC

The trained models exist inside the Kubernetes PVC (`stock-data` in the `kubeflow` namespace). To make them available to Docker Compose services, they were exported using a temporary busybox pod:

```bash
# Create temp pod with PVC mounted
kubectl apply -f pipeline/model-exporter-pod.yaml

# Wait for it to be ready
kubectl wait pod/model-exporter -n kubeflow --for=condition=Ready --timeout=60s

# Copy data out (run each separately - back-to-back kills API tunnel)
cd d:\Stock_Kubeflow
kubectl cp kubeflow/model-exporter:/data/models ./data/models
kubectl cp kubeflow/model-exporter:/data/combined ./data/combined
kubectl cp kubeflow/model-exporter:/data/features ./data/features

# Cleanup
kubectl delete pod model-exporter -n kubeflow
```

**Pod manifest** (`pipeline/model-exporter-pod.yaml`):

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: model-exporter
  namespace: kubeflow
spec:
  restartPolicy: Never
  volumes:
    - name: data
      persistentVolumeClaim:
        claimName: stock-data
  containers:
    - name: model-exporter
      image: busybox
      command: ["sleep", "300"]
      volumeMounts:
        - name: data
          mountPath: /data
```

**Problems encountered:**

| Problem | Cause | Fix |
|---|---|---|
| `Invalid JSON Patch` on `kubectl run --overrides` | PowerShell mangles single-quoted JSON | Use YAML file with `kubectl apply -f` |
| `one of src or dest must be local` with `d:\path` | Drive letter `d:` parsed as remote host spec | Use relative path `./data/models` from project root |
| `wsarecv: connection forcibly closed` on second `kubectl cp` | Two back-to-back cp commands overwhelm k3d API tunnel | Run each `kubectl cp` separately |

---

### 12.3 Step 2: Update docker-compose.yml

Changed from Docker named volumes to local bind-mounts so the exported data is visible to containers:

```yaml
# Before (named volume - isolated from host filesystem)
volumes:
  - stock_data:/data

# After (bind mount - directly reads from exported data)
volumes:
  - ./data:/data
```

This means FastAPI reads models from `d:\Stock_Kubeflow\data\models\` which is the same directory where `kubectl cp` deposited them.

---

### 12.4 Step 3: Fix FastAPI Feature Engineering Mismatch

The FastAPI `build_live_features()` function was originally written to fetch live data from yfinance and compute MA50/MA20. This had two problems:

1. **yfinance blocked** from Docker containers (same issue as in Kubeflow)
2. **Feature mismatch**: models were trained on MA10/MA20 but FastAPI computed MA50/MA20

**Fix**: Replaced the entire live-fetch function with a simple CSV read from the pipeline output:

```python
# Before (broken)
def build_live_features() -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=400)
    dfs = []
    for stock in STOCKS:
        raw = yf.download(stock, start=start, end=end, ...)  # BLOCKED
        ...
    # computed MA50 (wrong) and MA20

# After (correct)
def build_live_features() -> pd.DataFrame:
    features_path = os.path.join(DATA_DIR, "features", "stock_features.csv")
    if not os.path.exists(features_path):
        raise RuntimeError("Features file not found. Run the pipeline first.")
    df = pd.read_csv(features_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df
```

The `/history` endpoint was similarly fixed to read from the combined CSV instead of calling yfinance:

```python
# After
def history(stock: str, days: int = 90):
    combined_path = os.path.join(DATA_DIR, "combined", "all_10_stocks_10years.csv")
    df = pd.read_csv(combined_path, parse_dates=["Date"])
    close_col = f"{stock}_Close"
    cutoff = df["Date"].max() - timedelta(days=days)
    df = df[df["Date"] >= cutoff][["Date", close_col]].dropna()
    ...
```

---

### 12.5 Step 4: Fix Infrastructure Issues

**Grafana port conflict** — Port 3000 was already in use on the host:

```yaml
# Changed in docker-compose.yml
ports:
  - "3001:3000"   # host:container
environment:
  GF_SERVER_ROOT_URL: http://localhost:3001  # tell Grafana its own URL
```

**FastAPI healthcheck** — `curl` not available in `python:3.11-slim`:

```yaml
# Before (fails - curl not installed)
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]

# After (works - uses Python stdlib)
healthcheck:
  test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
```

---

### 12.6 Final Results

All 4 services running:

```
docker compose ps

NAME                STATUS
stock_fastapi       healthy    (port 8000)
stock_streamlit     running    (port 8501)
stock_prometheus    running    (port 9090)
stock_grafana       running    (port 3001)
```

**FastAPI** loads all 3 models on startup:
```
INFO: Loaded model: xgboost
INFO: Loaded model: random_forest
INFO: Loaded model: gradient_boosting
INFO: Application startup complete.
```

**Streamlit** Price History tab — shows AAPL close price chart from Jan to Apr 2026:
- Latest Close: $253.50
- Period High: $278.12
- Period Low: $246.63
- Period Return: -2.1%

**Streamlit** Prediction tab — XGBoost predicts AAPL close: **$253.51** for 2026-04-08

**Grafana** Model Metrics:
| Model | MSE | RMSE | R² |
|---|---|---|---|
| Gradient Boosting | 3.38 | 1.84 | 0.965 |
| Random Forest | 5.43 | 2.33 | 0.944 |
| XGBoost | 5.47 | 2.34 | 0.944 |

Gradient Boosting is the best performing model with R² = 0.965.

---

### 12.7 Service URLs

| Service | URL | Credentials |
|---|---|---|
| Streamlit Dashboard | http://localhost:8501 | None |
| FastAPI Swagger Docs | http://localhost:8000/docs | None |
| Prometheus | http://localhost:9090 | None |
| Grafana | http://localhost:3001 | admin / admin123 |
| Kubeflow UI | http://localhost:8080 | (port-forward required) |

---

*End of Report*

*This document captures the complete journey of building a production ML pipeline from scratch, including 3+ days of debugging Kubeflow installation issues, multiple API and SDK incompatibilities, container networking problems, database authentication challenges, and full serving layer integration. Every problem encountered had a specific root cause and a targeted fix — none were resolved by luck or random restarts.*

*Final system: 6-step Kubeflow pipeline (fetch > features > train x3 > evaluate) + 4 serving services (FastAPI + Streamlit + Prometheus + Grafana) fully connected and operational.*
