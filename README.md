# Stock Data Collection using yFinance

## Project Overview

This project downloads historical stock price data for 10 stocks over the last 10 years using the **yfinance** Python library.

The script automatically:

- Fetches daily stock price data
- Saves individual CSV files for each stock
- Creates one combined dataset
- Organizes files into a clean folder structure

This dataset can be used for:

- Data analysis
- Machine Learning models
- Time series forecasting
- Financial research

---

## Technologies Used

- Python
- yfinance
- pandas
- datetime

---

## Project Structure

stock_project/
│
├── data/
│ ├── raw/
│ │ ├── AAPL_10years.csv
│ │ ├── MSFT_10years.csv
│ │ ├── GOOGL_10years.csv
│ │ ├── AMZN_10years.csv
│ │ ├── TSLA_10years.csv
│ │ ├── META_10years.csv
│ │ ├── RELIANCE.NS_10years.csv
│ │ ├── TCS.NS_10years.csv
│ │ ├── INFY.NS_10years.csv
│ │ └── HDFCBANK.NS_10years.csv
│
│ └── combined/
│ └── all_10_stocks_10years.csv
│
├── src/
│ └── fetch_stock_data.py
│
├── requirements.txt
│
└── README.md

---

## Stock List

| Stock       | Company                   |
| ----------- | ------------------------- |
| AAPL        | Apple                     |
| MSFT        | Microsoft                 |
| GOOGL       | Google                    |
| AMZN        | Amazon                    |
| TSLA        | Tesla                     |
| META        | Meta                      |
| RELIANCE.NS | Reliance Industries       |
| TCS.NS      | Tata Consultancy Services |
| INFY.NS     | Infosys                   |
| HDFCBANK.NS | HDFC Bank                 |

---

## Data Description

Each dataset contains:

| Column    | Description             |
| --------- | ----------------------- |
| Date      | Trading date            |
| Open      | Opening price           |
| High      | Highest price           |
| Low       | Lowest price            |
| Close     | Closing price           |
| Adj Close | Adjusted closing price  |
| Volume    | Number of shares traded |

---

## Installation

Clone the repository:

git clone <your_repo_link>

Navigate to project folder:

cd stock_project

Install dependencies:

pip install -r requirements.txt

---

## How to Run

Navigate to src folder:

cd src

Run script:

python fetch_stock_data.py

---

## Output

### Individual files

data/raw/

Each stock has separate CSV file.

### Combined dataset

data/combined/all_10_stocks_10years.csv

Contains merged data of all stocks based on Date.

---

## Future Improvements

- Data visualization
- Feature engineering
- Stock price prediction using ML/DL models
- LSTM time series forecasting
- Correlation analysis between stocks

---

## Feature Engineering

This project includes a feature engineering module that computes technical indicators for each stock.

The following indicators are calculated:

- Moving Average (MA20, MA50)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)

These features are useful for machine learning models and financial analysis.

The script:

src/feature_engineering.py

Generates the dataset:

data/features/stock_features.csv

## Author

Vinit Khuman
