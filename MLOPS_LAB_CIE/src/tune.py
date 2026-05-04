import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import json
import os

df = pd.read_csv("../data/training_data.csv")

X = df.drop("watch_time_min", axis=1)
y = df["watch_time_min"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 7, 15],
    "min_samples_split": [2, 4]
}

mlflow.set_experiment("streamcast-watch-time-min")

with mlflow.start_run(run_name="tuning-streamcast") as parent_run:

    model = RandomForestRegressor(random_state=42)

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    preds = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    output = {
        "search_type": "grid",
        "n_folds": 5,
        "total_trials": len(grid.cv_results_["params"]),
        "best_params": grid.best_params_,
        "best_mae": mae,
        "best_cv_mae": -grid.best_score_,
        "parent_run_name": "tuning-streamcast"
    }

    os.makedirs("../results", exist_ok=True)

    with open("../results/step2_s2.json", "w") as f:
        json.dump(output, f, indent=4)

print("Task 2 completed ✅")