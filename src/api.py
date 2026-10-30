
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
from numbers import Real

import csv
import json
import math
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import RootModel


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
MONITORING_DIR = BASE_DIR / "monitoring"

PREDICTIONS_FILE = MONITORING_DIR / "predictions.csv"
PRODUCTION_INPUTS_FILE = MONITORING_DIR / "production_inputs.csv"
HTTP_LOG_FILE = MONITORING_DIR / "http_requests.jsonl"

MONITORING_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Chargement du modèle
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Modèle introuvable : {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

# Les colonnes attendues sont récupérées directement
# depuis le pipeline du modèle du Projet 6.
REQUIRED_FEATURES = list(model.feature_names_in_)


# ============================================================
# Application FastAPI
# ============================================================

app = FastAPI(
    title="Prêt à Dépenser - Credit Scoring API",
    description="API de prédiction du risque de défaut de paiement.",
    version="1.0.0",
)


# ============================================================
# Middleware de journalisation HTTP
# ============================================================

@app.middleware("http")
async def log_http_requests(request: Request, call_next):
    start = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response

    finally:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": request.method,
            "path": request.url.path,
            "status_code": status_code,
            "duration_ms": round(
                (time.perf_counter() - start) * 1000, 3
            ),
        }

        try:
            with HTTP_LOG_FILE.open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps(event, ensure_ascii=False) + "\n"
                )
        except OSError:
            # Une erreur de journalisation ne doit pas interrompre l'API.
            pass


# ============================================================
# Modèle de requête
# ============================================================

class PredictionRequest(RootModel[dict[str, Any]]):
    pass


# ============================================================
# Routes
# ============================================================

@app.get("/")
def root():
    return {
        "message": "API Credit Scoring - Prêt à Dépenser",
        "status": "ok",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "number_of_features": len(REQUIRED_FEATURES),
    }


@app.get("/model-info")
def model_info():
    return {
        "model_type": type(model).__name__,
        "number_of_features": len(REQUIRED_FEATURES),
        "required_features": REQUIRED_FEATURES,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    payload = request.root

    # --------------------------------------------------------
    # Vérification des variables nécessaires
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in payload
    ]

    if missing_features:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Certaines variables nécessaires au modèle sont absentes.",
                "missing_features": missing_features,
                "number_missing": len(missing_features),
            },
        )

    # --------------------------------------------------------
    # Préparation et validation des données
    # --------------------------------------------------------

    preprocessor = model.named_steps["preprocessor"]
    numeric_features = preprocessor.transformers_[0][2]
    categorical_features = preprocessor.transformers_[1][2]

    invalid_features = []

    for feature in numeric_features:
        value = payload[feature]

        if value is None:
            continue

        if isinstance(value, bool) or not isinstance(value, Real):
            invalid_features.append(feature)

        elif not math.isfinite(float(value)):
            invalid_features.append(feature)

    for feature in categorical_features:
        value = payload[feature]

        if value is not None and not isinstance(value, str):
            invalid_features.append(feature)

    if invalid_features:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Certaines variables ont un type ou une valeur invalide.",
                "invalid_features": invalid_features,
            },
        )

    data = pd.DataFrame([payload])

    # On conserve uniquement les variables utilisées
    # par le pipeline du modèle.
    data = data[REQUIRED_FEATURES]

    # --------------------------------------------------------
    # Prédiction et mesure du temps d'inférence
    # --------------------------------------------------------

    start_time = time.perf_counter()

    try:
        probability = float(model.predict_proba(data)[0, 1])

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Erreur pendant la prédiction : {str(exc)}",
        )

    inference_time_ms = (
        time.perf_counter() - start_time
    ) * 1000

    # --------------------------------------------------------
    # Enregistrement des variables pour le data drift
    # --------------------------------------------------------

    # L'identifiant client est exclu du journal.
    input_features = [
        feature
        for feature in REQUIRED_FEATURES
        if feature != "SK_ID_CURR"
    ]

    input_file_exists = PRODUCTION_INPUTS_FILE.exists()
    timestamp = datetime.now(timezone.utc).isoformat()

    with open(
        PRODUCTION_INPUTS_FILE,
        mode="a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        if not input_file_exists:
            writer.writerow(["timestamp", *input_features])

        writer.writerow([
            timestamp,
            *[
                data.iloc[0][feature]
                for feature in input_features
            ],
        ])

    # --------------------------------------------------------
    # Identification du client
    # --------------------------------------------------------

    customer_id = payload.get("SK_ID_CURR")

    # --------------------------------------------------------
    # Enregistrement des prédictions pour le monitoring
    # --------------------------------------------------------

    timestamp = datetime.now(timezone.utc).isoformat()
    file_exists = PREDICTIONS_FILE.exists()

    with open(
        PREDICTIONS_FILE,
        mode="a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "SK_ID_CURR",
                "score",
                "inference_time_ms",
            ])

        writer.writerow([
            timestamp,
            customer_id,
            probability,
            inference_time_ms,
        ])

    # --------------------------------------------------------
    # Réponse API
    # --------------------------------------------------------

    return {
        "SK_ID_CURR": customer_id,
        "score": round(probability, 6),
        "score_percent": round(probability * 100, 2),
        "inference_time_ms": round(inference_time_ms, 3),
    }
