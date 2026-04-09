# fetch_stock_data.py

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

# Step 1: stock list
stocks = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "META",
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS"
]

# Step 2: create folders
os.makedirs("./data/raw", exist_ok=True)
os.makedirs("./data/combined", exist_ok=True)

# Step 3: date range
end_date = datetime.today()
start_date = end_date - timedelta(days=365*10)

print("Fetching data from:", start_date.date(), "to", end_date.date())

# Step 4: list to store dfs
dfs = []

for stock in stocks:

    print(f"Downloading {stock}...")

    data = yf.download(
        stock,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False
    )

    data.reset_index(inplace=True)

    # save individual csv
    data.to_csv(f"./data/raw/{stock}_10years.csv", index=False)

    # rename columns with stock prefix
    data.columns = [
        "Date",
        f"{stock}_Open",
        f"{stock}_High",
        f"{stock}_Low",
        f"{stock}_Close",
        f"{stock}_AdjClose",
        f"{stock}_Volume"
    ]

    dfs.append(data)

# Step 5: merge all stocks on Date column
combined_df = dfs[0]

for df in dfs[1:]:
    combined_df = pd.merge(combined_df, df, on="Date", how="outer")

# Step 6: sort by date
combined_df = combined_df.sort_values("Date")

# Step 7: save combined csv
combined_df.to_csv("./data/combined/all_10_stocks_10years.csv", index=False)

print("\nCombined CSV saved successfully")
print(combined_df.head())