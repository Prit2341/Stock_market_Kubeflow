import pandas as pd
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

DATA_DIR = os.environ.get("DATA_DIR", "/data")
MODELS_DIR = os.environ.get("MODELS_DIR", "/data/models")

os.makedirs(MODELS_DIR, exist_ok=True)

features_path = os.path.join(DATA_DIR, "features", "stock_features.csv")
print(f"Loading dataset from {features_path}...")
df = pd.read_csv(features_path)

y = df["AAPL_Close"]
X = df.drop(columns=["Date", "AAPL_Close"])

# Drop rows with NaN (from TA indicator warmup period)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Gradient Boosting model...")
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
print(f"Gradient Boosting  MSE: {mse:.4f}  RMSE: {rmse:.4f}")

model_path = os.path.join(MODELS_DIR, "gradient_boosting_model.pkl")
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
