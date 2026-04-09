import pandas as pd
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

DATA_DIR = os.environ.get("DATA_DIR", "/data")
MODELS_DIR = os.environ.get("MODELS_DIR", "/data/models")

os.makedirs(MODELS_DIR, exist_ok=True)

features_path = os.path.join(DATA_DIR, "features", "stock_features.csv")
print(f"Loading dataset from {features_path}...")
df = pd.read_csv(features_path)

y = df["AAPL_Close"]
X = df.drop(columns=["Date", "AAPL_Close"])

mask = X.notna().all(axis=1) & y.notna()
X, y = X[mask], y[mask]
print(f"Dataset shape after dropping NaN: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training XGBoost model...")
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
print(f"XGBoost  MSE: {mse:.4f}  RMSE: {rmse:.4f}")

model_path = os.path.join(MODELS_DIR, "XGBoost_model.pkl")
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
