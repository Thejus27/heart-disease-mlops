import logging
import time

import joblib
import pandas as pd

from fastapi import FastAPI, Request
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, make_asgi_app


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("heart_disease_api")

app = FastAPI(title="Heart Disease Prediction API")

model = joblib.load("model/heart_disease_model.joblib")

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "API request duration in seconds",
    ["endpoint"]
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


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


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code
    ).inc()

    REQUEST_LATENCY.labels(
        endpoint=request.url.path
    ).observe(duration)

    logger.info(
        "%s %s - %s - %.4fs",
        request.method,
        request.url.path,
        response.status_code,
        duration
    )

    return response


@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API"}


@app.post("/predict")
def predict(data: PatientData):
    input_data = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data)[0][1]

    logger.info(
        "Prediction generated: class=%s confidence=%.4f",
        prediction,
        confidence
    )

    return {
        "prediction": int(prediction),
        "confidence": float(confidence)
    }
