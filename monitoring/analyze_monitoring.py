from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

BASE_DIR = Path(__file__).resolve().parent
REFERENCE_PATH = BASE_DIR / "reference_data.csv"
PREDICTIONS_PATH = BASE_DIR / "predictions.csv"
PRODUCTION_INPUTS_PATH = BASE_DIR / "production_inputs.csv"
REPORT_PATH = BASE_DIR / "monitoring_report.json"
MISSING_PATH = BASE_DIR / "missing_values_report.csv"
PLOTS_DIR = BASE_DIR / "reports"

MIN_PREDICTIONS = 30
DRIFT_P_VALUE_THRESHOLD = 0.05
DRIFT_FREQUENCY_THRESHOLD = 0.05


def analyze_data_drift(reference, production_inputs):
    """Compare les distributions de référence et de production."""

    results = []
    warnings = []

    if production_inputs is None:
        return {
            "status": "indisponible",
            "message": "Le fichier des variables de production est absent.",
            "features_tested": 0,
            "features_with_drift": 0,
            "features": [],
        }, warnings

    common_features = [
        feature
        for feature in reference.columns
        if feature in production_inputs.columns
    ]

    if not common_features:
        return {
            "status": "indisponible",
            "message": "Aucune variable commune entre référence et production.",
            "features_tested": 0,
            "features_with_drift": 0,
            "features": [],
        }, warnings

    if len(production_inputs) < MIN_PREDICTIONS:
        warnings.append(
            f"Data drift non évalué : {len(production_inputs)} observation(s) "
            f"disponible(s), contre {MIN_PREDICTIONS} minimum."
        )
        return {
            "status": "volume_insuffisant",
            "message": (
                "Le volume de production est trop faible pour une analyse "
                "statistique du drift."
            ),
            "production_rows": int(len(production_inputs)),
            "minimum_rows": MIN_PREDICTIONS,
            "features_compared": int(len(common_features)),
            "features_tested": 0,
            "features_with_drift": 0,
            "features": [],
        }, warnings

    for feature in common_features:
        ref_series = reference[feature]
        prod_series = production_inputs[feature]

        ref_numeric = pd.to_numeric(ref_series, errors="coerce")
        prod_numeric = pd.to_numeric(prod_series, errors="coerce")

        # Une variable est considérée numérique si ses valeurs non nulles
        # sont convertibles en nombres dans les deux jeux de données.
        ref_non_null = ref_series.dropna()
        prod_non_null = prod_series.dropna()

        is_numeric = (
            len(ref_non_null) > 0
            and len(prod_non_null) > 0
            and ref_numeric.notna().sum() / len(ref_non_null) >= 0.95
            and prod_numeric.notna().sum() / len(prod_non_null) >= 0.95
        )

        if is_numeric:
            ref_values = ref_numeric.replace([np.inf, -np.inf], np.nan).dropna()
            prod_values = prod_numeric.replace([np.inf, -np.inf], np.nan).dropna()

            if len(ref_values) < 2 or len(prod_values) < 2:
                continue

            statistic, p_value = ks_2samp(ref_values, prod_values)

            results.append({
                "variable": feature,
                "type": "numerique",
                "reference_count": int(len(ref_values)),
                "production_count": int(len(prod_values)),
                "ks_statistic": float(statistic),
                "p_value": float(p_value),
                "drift_detected": bool(p_value < DRIFT_P_VALUE_THRESHOLD),
            })

        else:
            ref_values = ref_series.dropna().astype(str)
            prod_values = prod_series.dropna().astype(str)

            if ref_values.empty or prod_values.empty:
                continue

            ref_freq = ref_values.value_counts(normalize=True)
            prod_freq = prod_values.value_counts(normalize=True)
            categories = ref_freq.index.union(prod_freq.index)

            max_difference = max(
                abs(
                    float(ref_freq.get(category, 0.0))
                    - float(prod_freq.get(category, 0.0))
                )
                for category in categories
            )

            results.append({
                "variable": feature,
                "type": "categorielle",
                "reference_count": int(len(ref_values)),
                "production_count": int(len(prod_values)),
                "max_frequency_difference": float(max_difference),
                "drift_detected": bool(
                    max_difference > DRIFT_FREQUENCY_THRESHOLD
                ),
            })

    drifted_features = [
        result for result in results if result["drift_detected"]
    ]

    return {
        "status": "analyse_effectuee",
        "message": (
            "Comparaison exploratoire des distributions. "
            "Les résultats ne prouvent pas une dégradation du modèle."
        ),
        "production_rows": int(len(production_inputs)),
        "features_compared": int(len(common_features)),
        "features_tested": int(len(results)),
        "features_with_drift": int(len(drifted_features)),
        "numeric_p_value_threshold": DRIFT_P_VALUE_THRESHOLD,
        "categorical_frequency_threshold": DRIFT_FREQUENCY_THRESHOLD,
        "features": results,
    }, warnings


def main():
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Jeu de référence introuvable : {REFERENCE_PATH}"
        )

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Journal des prédictions introuvable : {PREDICTIONS_PATH}"
        )

    reference = pd.read_csv(REFERENCE_PATH)
    production = pd.read_csv(PREDICTIONS_PATH)

    required = {"score", "inference_time_ms", "timestamp"}
    if not required.issubset(production.columns):
        missing_columns = required - set(production.columns)
        raise ValueError(
            f"Colonnes manquantes dans le journal : {missing_columns}"
        )

    if PRODUCTION_INPUTS_PATH.exists():
        production_inputs = pd.read_csv(PRODUCTION_INPUTS_PATH)
        if "timestamp" in production_inputs.columns:
            production_inputs = production_inputs.drop(columns=["timestamp"])
    else:
        production_inputs = None

    scores = pd.to_numeric(
        production["score"], errors="coerce"
    ).dropna()

    times = pd.to_numeric(
        production["inference_time_ms"], errors="coerce"
    ).dropna()

    missing_reference = reference.isna().mean().mul(100)
    missing_report = pd.DataFrame({
        "variable": missing_reference.index,
        "reference_missing_pct": missing_reference.values,
    })
    missing_report.to_csv(MISSING_PATH, index=False)

    enough_data = len(production) >= MIN_PREDICTIONS

    drift_report, drift_warnings = analyze_data_drift(
        reference, production_inputs
    )

    report = {
        "production_rows": int(len(production)),
        "minimum_rows_for_basic_analysis": MIN_PREDICTIONS,
        "enough_data_for_basic_analysis": bool(enough_data),
        "reference_rows": int(len(reference)),
        "reference_features": int(len(reference.columns)),
        "production_input_features_logged": production_inputs is not None,
        "production_input_rows": (
            int(len(production_inputs))
            if production_inputs is not None
            else 0
        ),
        "data_drift": drift_report,
        "score_summary": {},
        "inference_time_summary_ms": {},
        "operational_checks": {},
        "warnings": [],
    }

    report["warnings"].extend(drift_warnings)

    if len(scores):
        report["score_summary"] = {
            "count": int(len(scores)),
            "mean": float(scores.mean()),
            "median": float(scores.median()),
            "min": float(scores.min()),
            "max": float(scores.max()),
        }

        if ((scores < 0) | (scores > 1)).any():
            report["warnings"].append(
                "Des scores sont en dehors de l'intervalle [0, 1]."
            )

    if len(times):
        report["inference_time_summary_ms"] = {
            "count": int(len(times)),
            "mean": float(times.mean()),
            "median": float(times.median()),
            "p95": float(times.quantile(0.95)),
            "max": float(times.max()),
        }

    if not enough_data:
        report["warnings"].append(
            f"Seulement {len(production)} prédiction(s) : "
            f"volume inférieur au seuil de {MIN_PREDICTIONS}."
        )

    invalid_scores = (
        pd.to_numeric(production["score"], errors="coerce").isna()
        | ~pd.to_numeric(
            production["score"], errors="coerce"
        ).between(0, 1)
    )

    invalid_times = (
        pd.to_numeric(
            production["inference_time_ms"], errors="coerce"
        ).isna()
        | (pd.to_numeric(
            production["inference_time_ms"], errors="coerce"
        ) < 0)
    )

    report["operational_checks"] = {
        "prediction_rows": int(len(production)),
        "invalid_score_rows": int(invalid_scores.sum()),
        "invalid_inference_time_rows": int(invalid_times.sum()),
        "inference_time_over_1000ms_rows": int(
            (pd.to_numeric(
                production["inference_time_ms"], errors="coerce"
            ) > 1000).sum()
        ),
    }

    if report["operational_checks"]["inference_time_over_1000ms_rows"]:
        report["warnings"].append(
            "Certaines prédictions ont un temps d'inférence supérieur à 1000 ms."
        )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if len(scores):
        plt.figure(figsize=(8, 4))
        plt.hist(scores, bins=min(20, max(1, len(scores))))
        plt.title("Distribution des scores de production")
        plt.xlabel("Score prédit")
        plt.ylabel("Nombre de prédictions")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "production_scores.png", dpi=120)
        plt.close()

    if len(times):
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(1, len(times) + 1), times, marker="o")
        plt.title("Temps d'inférence par prédiction")
        plt.xlabel("Numéro de prédiction")
        plt.ylabel("Temps (ms)")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "inference_times.png", dpi=120)
        plt.close()

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Rapport JSON : {REPORT_PATH}")
    print(f"Rapport des valeurs manquantes : {MISSING_PATH}")
    print(f"Graphiques : {PLOTS_DIR}")
    print(f"Prédictions analysées : {len(production)}")
    print(f"Volume suffisant (seuil {MIN_PREDICTIONS}) : {enough_data}")
    print(f"État du data drift : {drift_report['status']}")
    print(
        "Variables avec drift détecté : "
        f"{drift_report['features_with_drift']}"
    )

    if report["warnings"]:
        print("Alertes :")
        for warning in report["warnings"]:
            print(f"- {warning}")


if __name__ == "__main__":
    main()