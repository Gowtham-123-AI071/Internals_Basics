import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import os
import joblib

# Load data
df = pd.read_csv("../data/training_data.csv")

X = df.drop("watch_time_min", axis=1)
y = df["watch_time_min"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("streamcast-watch-time-min")

results = []

def evaluate(model, name):
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_params(model.get_params())
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.set_tag("domain", "ott_streaming")

        mlflow.sklearn.log_model(model, name=name)

        return {
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "model_obj": model   # store model object
        }

# Models
lasso = Lasso()
rf = RandomForestRegressor(random_state=42)

results.append(evaluate(lasso, "Lasso"))
results.append(evaluate(rf, "RandomForest"))

# Select best model by RMSE
best_model = min(results, key=lambda x: x["rmse"])

# ✅ Save BEST model (IMPORTANT)
os.makedirs("../models", exist_ok=True)
joblib.dump(best_model["model_obj"], "../models/model.pkl")

# Remove model_obj before saving JSON (not serializable)
for r in results:
    r.pop("model_obj")

output = {
    "experiment_name": "streamcast-watch-time-min",
    "models": results,
    "best_model": best_model["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best_model["rmse"]
}

os.makedirs("../results", exist_ok=True)

with open("../results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task  completed ✅ and best model saved 🎯")