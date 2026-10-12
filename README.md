# Heart Disease MLOps

End-to-end MLOps project for binary heart disease prediction using the UCI Heart Disease dataset.

## Project Overview

This project includes:

- data acquisition and preprocessing
- exploratory data analysis
- Logistic Regression and Random Forest models
- cross-validation and model evaluation
- MLflow experiment tracking
- reusable model training pipeline
- FastAPI prediction API
- pytest unit tests
- GitHub Actions CI/CD
- Docker containerization
- Kubernetes deployment using Minikube
- Prometheus-compatible metrics
- request and prediction logging

## Dataset

The project uses the UCI Heart Disease dataset.

The dataset is downloaded programmatically with:

```python
from ucimlrepo import fetch_ucirepo

heart_disease = fetch_ucirepo(id=45)
```
A reproducible cleaning script is provided at:

src/prepare_data.py

Run:

python src/prepare_data.py

This creates:

data/heart_disease_clean.csv

The cleaned dataset contains 303 records, 13 features, one binary target, and no missing values.

Target encoding:
- 0 = no heart disease
- 1 = heart disease present

Continuous missing values are replaced using the median.
Categorical missing values are replaced using the mode.

## Setup

```bash
git clone https://github.com/Thejus27/heart-disease-mlops.git
cd heart-disease-mlops
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Training

Run:

python src/train.py

The trained model is saved to:

model/heart_disease_model.joblib

## Testing

Run:

python -m pytest -q

## MLflow

Start MLflow with:

mlflow ui --host 0.0.0.0 --port 5000

Then open:

http://127.0.0.1:5000

MLflow records model parameters, metrics, and artifacts for Logistic Regression and Random Forest runs.

## API

The FastAPI service provides:

POST /predict

Example request:

{
  "age": 63,
  "sex": 1,
  "cp": 1,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 2,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 3,
  "ca": 0,
  "thal": 6
}

The response contains the predicted class and confidence score.

## Docker

Build the image:

docker build -t heart-disease-api:latest .

Run the container:

docker run -d --name heart-disease-container -p 8000:8000 heart-disease-api:latest

Test the API:

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json

## Kubernetes

Start Minikube:

minikube start --driver=docker

Load the local Docker image:

minikube image load heart-disease-api:latest

Deploy the application:

minikube kubectl -- apply -f k8s/deployment.yaml
minikube kubectl -- apply -f k8s/service.yaml

Check resources:

minikube kubectl -- get deployments
minikube kubectl -- get pods
minikube kubectl -- get services

Get the service URL:

minikube service heart-disease-service --url

## Monitoring and Logging

Prometheus-compatible metrics are available at:

/metrics/

The API logs request status, request duration, prediction class, and prediction confidence.

View Kubernetes logs with:

minikube kubectl -- logs -l app=heart-disease-api --prefix=true --tail=100

## CI/CD

GitHub Actions performs:

- dependency installation
- linting
- unit testing
- model training
- trained model artifact upload

Workflow file:

.github/workflows/ci.yml

## Project Structure

heart-disease-mlops/
- .github/workflows/ci.yml
- data/heart_disease_clean.csv
- k8s/deployment.yaml
- k8s/service.yaml
- model/heart_disease_model.joblib
- screenshots/
- src/api.py
- src/prepare_data.py
- src/train.py
- tests/
- Assignment1.ipynb
- Dockerfile
- README.md
- requirements.txt
- sample_request.json

## Repository Access

This repository is intended to be public for assignment evaluation.

No GitHub token or secret is required to clone the public repository.
