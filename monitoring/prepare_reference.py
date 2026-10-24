import os
from pathlib import Path

import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "model" / "model.pkl"

# Chemin configurable vers les données sources.
# Définir la variable SOURCE_DATA_PATH sur chaque environnement.
SOURCE_PATH = Path(
    os.environ.get(
        "SOURCE_DATA_PATH",
        str(PROJECT_DIR / "data" / "model_dataset_train.csv"),
    )
)

OUTPUT_PATH = PROJECT_DIR / "monitoring" / "reference_data.csv"

SAMPLE_SIZE = 10_000
RANDOM_STATE = 42


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Données source introuvables : {SOURCE_PATH}. "
            "Définissez la variable SOURCE_DATA_PATH."
        )

    model = joblib.load(MODEL_PATH)
    required_features = list(model.feature_names_in_)

    print("Lecture des données de référence...")
    df = pd.read_csv(SOURCE_PATH, usecols=required_features)

    reference = df.sample(
        n=min(SAMPLE_SIZE, len(df)),
        random_state=RANDOM_STATE,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(OUTPUT_PATH, index=False)

    print(f"Fichier créé : {OUTPUT_PATH}")
    print(f"Nombre de lignes : {len(reference)}")
    print(f"Nombre de variables : {len(reference.columns)}")

    if "SK_ID_CURR" in reference.columns:
        print(
            "Nombre d'identifiants uniques : "
            f"{reference['SK_ID_CURR'].nunique()}"
        )

    print(f"Taille du fichier : {OUTPUT_PATH.stat().st_size / 1024:.1f} Ko")


if __name__ == "__main__":
    main()