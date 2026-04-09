# feature_engineering.py

import pandas as pd
import os
import ta

# ==========================================
# Step 1: File paths
# ==========================================

INPUT_FILE = "./data/combined/all_10_stocks_10years.csv"
OUTPUT_FOLDER = "./data/features"
OUTPUT_FILE = "./data/features/stock_features.csv"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# Step 2: Load dataset
# ==========================================

print("\nLoading combined dataset...")

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

print("Dataset loaded successfully")
print("Original Shape:", df.shape)

# ==========================================
# Step 3: Detect stock symbols
# ==========================================

stocks = set()

for col in df.columns:
    if "_Close" in col:
        stock = col.split("_")[0]
        stocks.add(stock)

stocks = sorted(list(stocks))

print("\nDetected stocks:")
print(stocks)

# ==========================================
# Step 4: Feature Engineering
# ==========================================

for stock in stocks:

    print(f"\nProcessing features for {stock}")

    close = df[f"{stock}_Close"]

    # Moving averages
    df[f"{stock}_MA20"] = close.rolling(window=20).mean()
    df[f"{stock}_MA50"] = close.rolling(window=50).mean()

    # RSI
    rsi = ta.momentum.RSIIndicator(close=close, window=14)
    df[f"{stock}_RSI"] = rsi.rsi()

    # MACD
    macd = ta.trend.MACD(close=close)

    df[f"{stock}_MACD"] = macd.macd()
    df[f"{stock}_MACD_signal"] = macd.macd_signal()
    df[f"{stock}_MACD_diff"] = macd.macd_diff()

# ==========================================
# Step 5: Handle Missing Values
# ==========================================

print("\nHandling missing values...")

# forward fill time-series gaps
df = df.ffill()

# backward fill remaining gaps
df = df.bfill()

# remove warm-up rows required for indicators
df = df.iloc[50:]

print("Missing values handled successfully")

# ==========================================
# Step 6: Save dataset
# ==========================================

df.to_csv(OUTPUT_FILE, index=False)

print("\nFeature Engineering Completed Successfully")
print("Saved dataset at:", OUTPUT_FILE)

print("\nFinal Dataset Shape:", df.shape)
print("\nSample Data:")
print(df.head())