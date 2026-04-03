import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

print("Loading dataset...")

df = pd.read_csv("data/features/stock_features.csv")

y = df["AAPL_Close"]
X = df.drop(["Date", "AAPL_Close"], axis=1)

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Random Forest model...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)

print("Random Forest MSE:", mse)

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/random_forest_model.pkl")

print("Random Forest model saved")