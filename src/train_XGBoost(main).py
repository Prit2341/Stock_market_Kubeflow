import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


print("Loading dataset...")

df = pd.read_csv("data/features/stock_features.csv")


# Target variable
y = df["AAPL_Close"]

# Features
X = df.drop(["Date", "AAPL_Close"], axis=1)


print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print("Training XGBoost model...")

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8
)

model.fit(X_train, y_train)


print("Evaluating model...")

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)

print("Mean Squared Error:", mse)


# Save model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/XGBoost_model.pkl")

print("Model saved successfully")