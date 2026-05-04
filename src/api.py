from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI()

# Dummy trained model load (replace after training)
model = joblib.load("../models/model.pkl")

class InputData(BaseModel):
    content_length_min: float = Field(..., ge=10, le=180)
    user_tenure_months: int = Field(..., ge=1, le=60)
    is_premium: int = Field(..., ge=0, le=1)
    genre_score: int = Field(..., ge=1, le=10)

@app.get("/heartbeat")
def heartbeat():
    return {"alive": True, "service": "StreamCast watch_time_min API"}

@app.post("/score")
def predict(data: InputData):
    features = np.array([[ 
        data.content_length_min,
        data.user_tenure_months,
        data.is_premium,
        data.genre_score
    ]])

    pred = model.predict(features)[0]

    return {"prediction": float(pred)}