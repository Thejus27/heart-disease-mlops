import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Heart Disease Prediction API")

model = joblib.load("model/heart_disease_model.joblib")


class PatientData(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: float


@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API"}


@app.post("/predict")
def predict(data: PatientData):
    input_data = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data)[0][1]

    return {
        "prediction": int(prediction),
        "confidence": float(confidence)
    }
