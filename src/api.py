from pathlib import Path
from typing import Any
from datetime import datetime, timezone
import csv
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import RootModel


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
MONITORING_DIR = BASE_DIR / "monitoring"
PREDICTIONS_FILE = MONITORING_DIR / "predictions.csv"

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
    # Préparation des données
    # --------------------------------------------------------

    data = pd.DataFrame([payload])

    # On conserve uniquement les variables utilisées
    # par le pipeline du modèle.
    data = data[REQUIRED_FEATURES]

    # --------------------------------------------------------
    # Prédiction + mesure du temps d'inférence
    # --------------------------------------------------------

    start_time = time.perf_counter()

    try:
        probability = float(model.predict_proba(data)[0, 1])
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Erreur pendant la prédiction : {str(exc)}",
        )

    inference_time_ms = (time.perf_counter() - start_time) * 1000

    # --------------------------------------------------------
    # Identification du client
    # --------------------------------------------------------

    customer_id = payload.get("SK_ID_CURR")

    # --------------------------------------------------------
    # Enregistrement pour le monitoring
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
            writer.writerow(
                [
                    "timestamp",
                    "SK_ID_CURR",
                    "score",
                    "inference_time_ms",
                ]
            )

        writer.writerow(
            [
                timestamp,
                customer_id,
                probability,
                inference_time_ms,
            ]
        )

    # --------------------------------------------------------
    # Réponse API
    # --------------------------------------------------------

    return {
        "SK_ID_CURR": customer_id,
        "score": round(probability, 6),
        "score_percent": round(probability * 100, 2),
        "inference_time_ms": round(inference_time_ms, 3),
    }