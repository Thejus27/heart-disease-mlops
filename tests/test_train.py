from src.train import load_data, build_pipeline, train_model


def test_load_data():
    data = load_data()

    assert data.shape[0] > 0
    assert "target" in data.columns
    assert "num" not in data.columns
    assert set(data["target"].unique()).issubset({0, 1})


def test_build_pipeline():
    model = build_pipeline()

    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps


def test_train_model():
    model, X_test, y_test = train_model()

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)
    assert set(predictions).issubset({0, 1})
