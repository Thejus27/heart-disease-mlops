import joblib
import pandas as pd
from pathlib import Path

from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]

NUMERICAL_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]


def load_data():
    heart_disease = fetch_ucirepo(id=45)

    features = heart_disease.data.features.copy()
    target = heart_disease.data.targets.copy()

    data = pd.concat([features, target], axis=1)
    data["target"] = (data["num"] > 0).astype(int)
    data = data.drop(columns=["num"])

    return data


def build_pipeline():
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=1,
            max_iter=1000,
            random_state=42
        ))
    ])

    return model


def train_model():
    data = load_data()

    X = data.drop(columns=["target"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    return model, X_test, y_test


if __name__ == "__main__":
    model, _, _ = train_model()

    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)

    joblib.dump(model, model_dir / "heart_disease_model.joblib")

    print("Training completed successfully.")
