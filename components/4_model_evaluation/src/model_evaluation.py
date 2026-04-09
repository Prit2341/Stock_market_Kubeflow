import pandas as pd
import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

DATA_DIR = os.environ.get("DATA_DIR", "/data")
MODELS_DIR = os.environ.get("MODELS_DIR", "/data/models")

features_path = os.path.join(DATA_DIR, "features", "stock_features.csv")
print(f"Loading dataset from {features_path}...")
df = pd.read_csv(features_path)

y = df["AAPL_Close"]
X = df.drop(columns=["Date", "AAPL_Close"])

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

MODEL_FILES = {
    "XGBoost": "XGBoost_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "Gradient Boosting": "gradient_boosting_model.pkl",
}

results = []

for name, filename in MODEL_FILES.items():
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, skipping {name}.")
        continue

    print(f"Evaluating {name}...")
    model = joblib.load(path)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    results.append({"Model": name, "MSE": mse, "RMSE": rmse, "R2": r2})
    print(f"  MSE={mse:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

if not results:
    raise RuntimeError("No models were evaluated. Run training steps first.")

results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df.to_string(index=False))

# Save full results
all_path = os.path.join(MODELS_DIR, "all_models_results.csv")
results_df.to_csv(all_path, index=False)
print(f"\nAll results saved to {all_path}")

# Save best model row (fix: use DataFrame slice, not Series)
best_idx = results_df["MSE"].idxmin()
best_df = results_df.loc[[best_idx]]
best_path = os.path.join(MODELS_DIR, "best_model.csv")
best_df.to_csv(best_path, index=False)
print(f"Best model: {best_df['Model'].values[0]} (MSE={best_df['MSE'].values[0]:.4f})")
print(f"Best model info saved to {best_path}")
