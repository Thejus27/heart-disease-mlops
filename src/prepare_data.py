from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]


def prepare_data():
    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features.copy()
    y = heart_disease.data.targets.copy()

    df = pd.concat([X, y], axis=1)

    target_col = y.columns[0]
    df["target"] = (
        pd.to_numeric(df[target_col], errors="coerce") > 0
    ).astype(int)

    if target_col != "target":
        df = df.drop(columns=[target_col])

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


if __name__ == "__main__":
    output_path = Path("data/heart_disease_clean.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = prepare_data()
    df.to_csv(output_path, index=False)

    print(f"Saved cleaned dataset to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
