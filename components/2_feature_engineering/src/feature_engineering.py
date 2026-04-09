import pandas as pd
import os
import ta

DATA_DIR = os.environ.get("DATA_DIR", "/data")
INPUT_FILE = os.path.join(DATA_DIR, "combined", "all_10_stocks_10years.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "features")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "stock_features.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading combined dataset...")
df = pd.read_csv(INPUT_FILE)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
print(f"Loaded shape: {df.shape}")

# Detect stock symbols — only columns that end with "_Close"
# (avoids matching "_AdjClose")
stocks = sorted({
    col.rsplit("_Close", 1)[0]
    for col in df.columns
    if col.endswith("_Close") and not col.endswith("_AdjClose")
})
print(f"Detected stocks: {stocks}")

for stock in stocks:
    close_col = f"{stock}_Close"
    if close_col not in df.columns:
        print(f"WARNING: {close_col} not found, skipping.")
        continue

    print(f"  Engineering features for {stock}...")
    close = df[close_col].astype(float)

    df[f"{stock}_MA10"] = close.rolling(window=10).mean()
    df[f"{stock}_MA20"] = close.rolling(window=20).mean()

    df[f"{stock}_RSI"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df[f"{stock}_MACD"] = macd.macd()
    df[f"{stock}_MACD_signal"] = macd.macd_signal()
    df[f"{stock}_MACD_diff"] = macd.macd_diff()

# Fill NaNs from indicator warmup, then drop remaining NaN rows
df = df.ffill().bfill().dropna().reset_index(drop=True)

df.to_csv(OUTPUT_FILE, index=False)
print(f"\nFeatures saved to {OUTPUT_FILE}")
print(f"Final shape: {df.shape}")
