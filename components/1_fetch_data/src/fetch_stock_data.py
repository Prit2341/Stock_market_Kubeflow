import pandas as pd
import requests
import os
import time
from datetime import datetime

DATA_DIR   = os.environ.get("DATA_DIR", "/data")
RAW_DIR    = os.path.join(DATA_DIR, "raw")
COMBINED_DIR = os.path.join(DATA_DIR, "combined")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(COMBINED_DIR, exist_ok=True)

# Alpha Vantage free API — get key at https://www.alphavantage.co/support/#api-key
API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "demo")
BASE_URL = "https://www.alphavantage.co/query"

STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META",
    "RELIANCE.BSE", "TCS.BSE", "INFY", "HDB"
]

out_path = os.path.join(COMBINED_DIR, "all_10_stocks_10years.csv")

if os.path.exists(out_path):
    print(f"Data already exists at {out_path}, skipping download.")
    import sys; sys.exit(0)


def fetch_daily(symbol: str) -> pd.DataFrame:
    """Fetch full daily OHLCV history from Alpha Vantage."""
    params = {
        "function":   "TIME_SERIES_DAILY",
        "symbol":     symbol,
        "outputsize": "compact",    # last 100 days (free tier); "full" is premium
        "datatype":   "csv",
        "apikey":     API_KEY,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()

    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))

    if "timestamp" not in df.columns:
        raise ValueError(f"Unexpected response for {symbol}: {resp.text[:200]}")

    df.rename(columns={
        "timestamp": "Date",
        "open":      f"{symbol}_Open",
        "high":      f"{symbol}_High",
        "low":       f"{symbol}_Low",
        "close":     f"{symbol}_Close",
        "volume":    f"{symbol}_Volume",
    }, inplace=True)
    df.drop(columns=["adjusted close"], errors="ignore", inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


print(f"Fetching data using Alpha Vantage API (key={'demo' if API_KEY == 'demo' else 'set'})")

dfs = []
for stock in STOCKS:
    print(f"Downloading {stock}...")
    for attempt in range(1, 4):
        try:
            df = fetch_daily(stock)
            df.to_csv(os.path.join(RAW_DIR, f"{stock}_history.csv"), index=False)
            dfs.append(df)
            print(f"  OK — {len(df)} rows")
            time.sleep(15)   # Alpha Vantage free tier: max 25 req/day, ~1 req/sec burst
            break
        except Exception as e:
            print(f"  attempt {attempt} failed: {e}")
            if attempt < 3:
                time.sleep(15)

if not dfs:
    raise RuntimeError("No stock data was fetched. Check ALPHA_VANTAGE_API_KEY.")

combined_df = dfs[0]
for df in dfs[1:]:
    combined_df = pd.merge(combined_df, df, on="Date", how="outer")

combined_df = combined_df.sort_values("Date").reset_index(drop=True)
combined_df.to_csv(out_path, index=False)

print(f"\nCombined CSV saved to {out_path}")
print(f"Shape: {combined_df.shape}")
