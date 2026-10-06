import json
from pathlib import Path

import joblib
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
OUTPUT_PATH = BASE_DIR / "tests" / "test_payload.json"


# ============================================================
# Chargement du modèle
# ============================================================

model = joblib.load(MODEL_PATH)

features = list(model.feature_names_in_)

preprocessor = model.named_steps["preprocessor"]


# ============================================================
# Recherche automatique des variables catégorielles
# et du OneHotEncoder
# ============================================================

categorical_features = []
encoder = None

for name, transformer, columns in preprocessor.transformers_:

    if name == "cat":
        categorical_features = list(columns)

        # Recherche du OneHotEncoder dans le pipeline
        if hasattr(transformer, "named_steps"):
            for step_name, step in transformer.named_steps.items():
                if isinstance(step, OneHotEncoder):
                    encoder = step
                    break

        break


if encoder is None:
    raise RuntimeError(
        "Impossible de trouver le OneHotEncoder dans le préprocesseur."
    )


# ============================================================
# Création du payload de test
# ============================================================

payload = {}

for feature in features:

    if feature in categorical_features:

        index = categorical_features.index(feature)

        categories = encoder.categories_[index]

        if len(categories) == 0:
            raise RuntimeError(
                f"Aucune catégorie disponible pour {feature}"
            )

        # Première catégorie connue par le modèle
        payload[feature] = str(categories[0])

    else:
        # Valeur numérique neutre pour le test technique
        payload[feature] = 0


# ============================================================
# Identifiant client
# ============================================================

if "SK_ID_CURR" in payload:
    payload["SK_ID_CURR"] = 100001


# ============================================================
# Sauvegarde
# ============================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        payload,
        file,
        indent=2,
        ensure_ascii=False
    )


print("========================================")
print("Payload de test créé avec succès")
print("========================================")
print(f"Fichier : {OUTPUT_PATH}")
print(f"Nombre de variables : {len(payload)}")
print(f"Variables catégorielles : {len(categorical_features)}")
print("========================================")