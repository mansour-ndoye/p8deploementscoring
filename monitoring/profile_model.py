import json
import time
import cProfile
import pstats
import io
import joblib
import pandas as pd
from pathlib import Path
import psutil

MODEL_PATH = Path("model/model.pkl")
PAYLOAD_PATH = Path("tests/test_payload.json")
OUTPUT_PATH = Path("monitoring/profile_results.txt")

# Chargement du modèle et des données
model = joblib.load(MODEL_PATH)

with open(PAYLOAD_PATH, encoding="utf-8") as f:
    payload = json.load(f)

X = pd.DataFrame([payload])

# Vérification de la prédiction
prediction = model.predict_proba(X)[:, 1][0]
print(f"Score de référence : {prediction:.6f}")

# Échauffement
for _ in range(10):
    model.predict_proba(X)

# Mesure de 100 prédictions
durations = []

for _ in range(100):
    start = time.perf_counter()
    model.predict_proba(X)
    durations.append((time.perf_counter() - start) * 1000)

print(f"Temps moyen : {sum(durations) / len(durations):.3f} ms")
print(f"Temps minimum : {min(durations):.3f} ms")
print(f"Temps maximum : {max(durations):.3f} ms")

# Mesure CPU et mémoire
process = psutil.Process()
memory_before = process.memory_info().rss / (1024 * 1024)
cpu_before = process.cpu_percent(interval=1)

for _ in range(100):
    model.predict_proba(X)

cpu_after = process.cpu_percent(interval=1)
memory_after = process.memory_info().rss / (1024 * 1024)

print(f"CPU utilisé : {cpu_after:.2f} %")
print(f"Mémoire avant : {memory_before:.2f} Mo")
print(f"Mémoire après : {memory_after:.2f} Mo")

# Profiling détaillé
profiler = cProfile.Profile()
profiler.enable()

for _ in range(100):
    model.predict_proba(X)

profiler.disable()

output = io.StringIO()
stats = pstats.Stats(profiler, stream=output)
stats.strip_dirs().sort_stats("cumtime").print_stats(20)

OUTPUT_PATH.write_text(output.getvalue(), encoding="utf-8")

print(f"\nProfiling enregistré dans : {OUTPUT_PATH}")
print("\nTop 20 des opérations les plus coûteuses :")
print(output.getvalue())
# Comparaison : pipeline complet vs exécution séparée
import numpy as np

# Prétraitement effectué une seule fois
X_transformed = model.named_steps["preprocessor"].transform(X)
classifier = model.named_steps["classifier"]

# Vérification de l'égalité des scores
score_pipeline = model.predict_proba(X)[0, 1]
score_direct = classifier.predict_proba(X_transformed)[0, 1]

print("\n--- Comparaison des prédictions ---")
print(f"Score pipeline : {score_pipeline:.12f}")
print(f"Score direct : {score_direct:.12f}")
print(f"Scores identiques : {np.isclose(score_pipeline, score_direct)}")

# Mesure de la classification seule sur les données déjà transformées
for _ in range(10):
    classifier.predict_proba(X_transformed)

direct_durations = []

for _ in range(100):
    start = time.perf_counter()
    classifier.predict_proba(X_transformed)
    direct_durations.append((time.perf_counter() - start) * 1000)

print(
    f"Classification seule : "
    f"{sum(direct_durations) / len(direct_durations):.3f} ms"
)

# Benchmark comparatif sur 500 prédictions
print("\n--- Benchmark comparatif ---")

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# Échauffement
for _ in range(20):
    model.predict_proba(X)
    Z = preprocessor.transform(X)
    classifier.predict_proba(Z)

# Pipeline complet
start = time.perf_counter()
for _ in range(500):
    model.predict_proba(X)
pipeline_time = (time.perf_counter() - start) * 1000 / 500

# Prétraitement + classification séparés
start = time.perf_counter()
for _ in range(500):
    Z = preprocessor.transform(X)
    classifier.predict_proba(Z)
direct_time = (time.perf_counter() - start) * 1000 / 500

print(f"Pipeline complet : {pipeline_time:.3f} ms")
print(f"Exécution séparée : {direct_time:.3f} ms")
print(f"Gain : {pipeline_time - direct_time:.3f} ms")
print(
    f"Amélioration : "
    f"{(pipeline_time - direct_time) / pipeline_time * 100:.2f} %"
)