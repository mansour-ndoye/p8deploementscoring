import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app


BASE_DIR = Path(__file__).resolve().parents[1]
PAYLOAD_PATH = BASE_DIR / "tests" / "test_payload.json"


client = TestClient(app)


def load_payload():
    with open(PAYLOAD_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["number_of_features"] == 304


def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_type"] == "Pipeline"
    assert data["number_of_features"] == 304
    assert "SK_ID_CURR" in data["required_features"]


def test_predict():
    payload = load_payload()

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert data["SK_ID_CURR"] == 100001

    assert "score" in data
    assert "score_percent" in data
    assert "inference_time_ms" in data

    assert 0 <= data["score"] <= 1
    assert 0 <= data["score_percent"] <= 100
    assert data["inference_time_ms"] >= 0


def test_predict_missing_feature():
    payload = load_payload()

    # Suppression volontaire d'une variable obligatoire
    payload.pop("AMT_INCOME_TOTAL")

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422

    data = response.json()

    assert "missing_features" in data["detail"]
    assert "AMT_INCOME_TOTAL" in data["detail"]["missing_features"]
def test_predict_invalid_numeric_type():
    payload = load_payload()
    payload["AMT_INCOME_TOTAL"] = "texte_invalide"

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422

    data = response.json()

    assert "invalid_features" in data["detail"]
    assert "AMT_INCOME_TOTAL" in data["detail"]["invalid_features"]

def test_predict_invalid_categorical_type():
    payload = load_payload()
    payload["NAME_CONTRACT_TYPE"] = 123

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422

    data = response.json()

    assert "invalid_features" in data["detail"]
    assert "NAME_CONTRACT_TYPE" in data["detail"]["invalid_features"]

def test_predict_infinite_numeric_value():
    payload = load_payload()
    payload["AMT_INCOME_TOTAL"] = float("inf")

    import json

    response = client.post(
        "/predict",
        content=json.dumps(payload, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422

    data = response.json()

    assert "invalid_features" in data["detail"]
    assert "AMT_INCOME_TOTAL" in data["detail"]["invalid_features"]
