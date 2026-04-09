import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt
import os

print("Loading dataset...")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

file_path = os.path.join(BASE_DIR, "data", "features", "stock_features.csv")

print("Reading from:", file_path)

df = pd.read_csv(file_path)

y = df["AAPL_Close"]
X = df.drop(["Date", "AAPL_Close"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_dir = os.path.join(BASE_DIR, "models")

models = {
    "XGBoost": joblib.load(os.path.join(model_dir, "XGBoost_model.pkl")),
    "Random Forest": joblib.load(os.path.join(model_dir, "random_forest_model.pkl")),
    "Gradient Boosting": joblib.load(os.path.join(model_dir, "gradient_boosting_model.pkl"))
}
results = []

for name, model in models.items():

    print(f"\nEvaluating {name}...")

    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    results.append([name, mse, rmse, r2])

    print(f"{name} Results:")
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)


# Final comparison table
results_df = pd.DataFrame(results, columns=["Model", "MSE", "RMSE", "R2"])

print("\nFinal Model Comparison:")
print(results_df)

# Best model
best_model = results_df.loc[results_df["MSE"].idxmin()]

print("\nBest Model:")
print(best_model)

print("\nBest Model Selected:", best_model["Model"])
best_model.to_csv("models/best_model.csv", index=False)