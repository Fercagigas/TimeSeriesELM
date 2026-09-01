import json
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from TimeSeriesELM import TimeSeriesELM


def _build_dataset(n_samples: int = 240, lookback: int = 8):
    x = np.linspace(0, 10, n_samples)
    signal = np.sin(x) + 0.1 * np.cos(2 * x)
    X, y = [], []
    for i in range(n_samples - lookback):
        X.append(signal[i : i + lookback])
        y.append(signal[i + lookback])
    return np.array(X), np.array(y)


def test_fit_predict_shape_and_metrics():
    X, y = _build_dataset()
    model = TimeSeriesELM(n_hidden=32, activation="tanh", random_state=7)

    model.fit(X, y)
    preds = model.predict(X[:20])

    assert preds.shape == (20,)
    assert model.training_metrics_["rmse"] >= 0
    assert -1.0 <= model.training_metrics_["r2"] <= 1.0


def test_model_roundtrip_keeps_predictions(tmp_path: Path):
    X, y = _build_dataset()
    model = TimeSeriesELM(n_hidden=40, activation="relu", random_state=11, scale_data=True)
    model.fit(X, y)

    original_pred = model.predict(X[:15])
    model_path = tmp_path / "elm_model.json"
    model.save_model(model_path)

    restored = TimeSeriesELM()
    restored.load_model(model_path)
    restored_pred = restored.predict(X[:15])

    np.testing.assert_allclose(original_pred, restored_pred, atol=1e-10)

    with model_path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    assert "scaler_state" in state


def test_predict_rejects_wrong_feature_count():
    X, y = _build_dataset(lookback=5)
    model = TimeSeriesELM(n_hidden=20, random_state=1)
    model.fit(X, y)

    wrong_shape = np.ones((10, 3))
    try:
        model.predict(wrong_shape)
    except ValueError as exc:
        assert "features" in str(exc)
    else:
        raise AssertionError("Expected ValueError for wrong number of features")
