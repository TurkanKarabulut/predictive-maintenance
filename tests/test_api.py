import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import app


ROOT = Path(__file__).resolve().parents[1]

SAMPLE = {
    "type": "M",
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1551,
    "torque": 42.8,
    "tool_wear": 0,
}


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_prediction_matches_saved_model(client):
    metadata = json.loads(
        (ROOT / "models" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    model = joblib.load(
        ROOT / "models" / metadata["artifact"]
    )
    row = pd.DataFrame([{
        "Type": SAMPLE["type"],
        "Air temperature [K]": SAMPLE["air_temperature"],
        "Process temperature [K]": SAMPLE["process_temperature"],
        "Rotational speed [rpm]": SAMPLE["rotational_speed"],
        "Torque [Nm]": SAMPLE["torque"],
        "Tool wear [min]": SAMPLE["tool_wear"],
    }], columns=metadata["features"])

    expected_probability = float(
        model.predict_proba(row)[0, 1]
    )

    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    result = response.json()

    expected_prediction = int(
        expected_probability >= metadata["decision_threshold"]
    )
    assert result["probability"] == pytest.approx(
        expected_probability
    )
    assert result["prediction"] == expected_prediction
    assert result["status"] == (
        "Fault" if expected_prediction else "Normal"
    )
    assert result["model_version"] == metadata["model_version"]


@pytest.mark.parametrize("change", [
    {"type": "INVALID"},
    {"air_temperature": -10},
    {"process_temperature": 0},
    {"rotational_speed": 0},
    {"torque": -1},
    {"tool_wear": -1},
    {"unexpected_field": 123},
])
def test_invalid_input_is_rejected(client, change):
    response = client.post(
        "/predict",
        json={**SAMPLE, **change},
    )
    assert response.status_code == 422


def test_missing_field_is_rejected(client):
    payload = SAMPLE.copy()
    del payload["torque"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_repeated_prediction_is_consistent(client):
    first = client.post("/predict", json=SAMPLE).json()
    second = client.post("/predict", json=SAMPLE).json()

    for field in [
        "prediction",
        "status",
        "probability",
        "model_version",
    ]:
        assert first[field] == second[field]

def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'id="prediction-form"' in response.text