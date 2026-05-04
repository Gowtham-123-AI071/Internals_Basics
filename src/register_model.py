import mlflow
import json
import os

RUN_ID = "c881d56417ab41cbbaa15dde4b7c70ad"
MODEL_NAME = "RandomForest"   # change to "Lasso" only if that was best

model_uri = f"runs:/{RUN_ID}/{MODEL_NAME}"

result = mlflow.register_model(
    model_uri,
    "streamcast-watch-time-min-predictor"
)

output = {
    "registered_model_name": "streamcast-watch-time-min-predictor",
    "version": result.version,
    "run_id": RUN_ID,
    "source_metric": "rmse",
    "source_metric_value": 0.0
}

os.makedirs("../results", exist_ok=True)

with open("../results/step4_s6.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 4 completed ✅")