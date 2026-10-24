Prêt à Dépenser — API de scoring crédit
1. Présentation du projet

Ce projet s’inscrit dans le cadre du Projet 8 de la formation Data Scientist d’OpenClassrooms.

L’objectif est de mettre à disposition un modèle de scoring crédit développé lors du Projet 6, à travers une API permettant de réaliser des prédictions à partir des données d’un client.

Le projet comprend :

Un modèle de machine learning sauvegardé au format pickle.
Une API REST développée avec FastAPI.
Des tests automatisés.
Une conteneurisation avec Docker.
Un pipeline CI/CD avec GitHub Actions.
2. Architecture du projet
p8deploementscoring/
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── model/
│   └── model.pkl
├── monitoring/
├── src/
│   └── api.py
├── tests/
│   ├── test_api.py
│   └── test_payload.json
├── Dockerfile
├── requirements.txt
└── README.md
3. Technologies utilisées
Python 3.13
FastAPI
Scikit-learn
LightGBM
Docker
Pytest
GitHub Actions
4. Installation et lancement local
Cloner le dépôt
git clone https://github.com/mansour-ndoye/p8deploementscoring.git
cd p8deploementscoring
Créer un environnement virtuel

Sous Windows :

python -m venv .venv
.\.venv\Scripts\Activate.ps1
Installer les dépendances
python -m pip install -r requirements.txt
Lancer l'API
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000

L'API est alors accessible à l'adresse :

http://127.0.0.1:8000

Documentation interactive :

http://127.0.0.1:8000/docs

5. Endpoints de l'API
Endpoint	Méthode	Description
/	GET	Point d'entrée de l'API
/health	GET	Vérification de l'état de santé de l'API
/model-info	GET	Informations sur le modèle chargé
/predict	POST	Réalisation d'une prédiction

L'endpoint /predict attend un objet JSON contenant les variables requises par le modèle.

Un exemple de requête est disponible dans tests/test_payload.json.

6. Lancer les tests

Pour exécuter les tests automatisés :

python -m pytest -v

Les tests couvrent notamment les endpoints principaux, les informations du modèle, la prédiction et la gestion d'une variable manquante.

7. Utilisation de Docker
Construire l'image
docker build -t pret-a-depenser-api:1.0 .
Démarrer le conteneur
docker run -d --name pret-a-depenser-api -p 8001:8000 pret-a-depenser-api:1.0
Vérifier la santé de l'API
Invoke-RestMethod http://127.0.0.1:8001/health
Tester une prédiction
$payload = Get-Content .\tests\test_payload.json -Raw

Invoke-RestMethod `
  -Uri http://127.0.0.1:8001/predict `
  -Method Post `
  -ContentType "application/json" `
  -Body $payload
Arrêter et supprimer le conteneur
docker rm -f pret-a-depenser-api
8. Pipeline CI/CD

Le workflow GitHub Actions se trouve dans .github/workflows/ci-cd.yml.

Il comporte deux jobs :

Automated tests : installe les dépendances Python et exécute les tests avec Pytest.
Docker build and deployment smoke test : après la réussite des tests, construit l'image Docker, démarre le conteneur et vérifie les endpoints /health et /predict.

Le pipeline est déclenché lors d'un push sur la branche main ou d'une pull request ciblant cette branche.

Le déploiement réalisé dans ce workflow est un déploiement de validation temporaire sur un runner GitHub Actions. Il ne correspond pas à une mise en production permanente.

9. Dépôt GitHub

Code source et workflow CI/CD :

https://github.com/mansour-ndoye/p8deploementscoring

10. Monitoring et surveillance du modèle

Stockage des données

L'API enregistre localement les informations nécessaires au suivi des prédictions dans le dossier monitoring/ :

predictions.csv : date de prédiction, identifiant client, score et temps d'inférence.

production_inputs.csv : variables d'entrée utilisées lors des prédictions.

reference_data.csv : échantillon de référence issu des données d'entraînement, utilisé pour comparer la distribution des variables.

Ces fichiers peuvent contenir des données sensibles. Ils sont destinés à un usage local et ne doivent pas être publiés sur le dépôt GitHub. Les fichiers de données et les rapports générés sont exclus du suivi Git et du contexte de construction Docker.

Cette solution constitue une preuve de concept locale. En production réelle, il faudrait prévoir un stockage sécurisé, des contrôles d'accès, une durée de conservation définie et une politique de minimisation des données personnelles.

Préparation des données de référence

Le script monitoring/prepare_reference.py prépare un échantillon de référence de 10 000 lignes à partir du jeu de données d'entraînement.

Par défaut, le script recherche les données dans data/model_dataset_train.csv. Un autre chemin peut être fourni via la variable d'environnement SOURCE_DATA_PATH.

Exemple sous PowerShell :

$env:SOURCE_DATA_PATH = "C:\chemin\vers\model_dataset_train.csv"
python .\monitoring\prepare_reference.py

Le fichier monitoring/reference_data.csv est généré localement et ne doit pas être partagé publiquement.

Analyse automatisée

Le script monitoring/analyze_monitoring.py produit un rapport de surveillance comprenant :

le volume de prédictions disponibles ;

les valeurs manquantes dans les variables de production ;

des statistiques sur les scores et les temps d'inférence ;

des contrôles opérationnels, notamment sur les scores invalides et les temps d'inférence supérieurs à 1 000 ms ;

une comparaison des distributions entre les données de référence et les données de production.

La détection de dérive utilise le test de Kolmogorov-Smirnov pour les variables numériques et une comparaison de fréquences pour les variables catégorielles.

Pour lancer l'analyse :

python .\monitoring\analyze_monitoring.py

Les résultats sont enregistrés localement sous forme de rapport JSON, de tableau CSV sur les valeurs manquantes et de graphiques dans monitoring/reports/.

Interprétation et limites

Le seuil minimal configuré est de 30 observations de production. En dessous de ce seuil, le rapport indique volume_insuffisant et la dérive n'est pas considérée comme évaluée.

Ce seuil est un choix opérationnel pour la preuve de concept ; il ne garantit pas une puissance statistique suffisante. Les résultats doivent être interprétés en tenant compte du volume, de la qualité des données et de la représentativité de l'échantillon.

Une dérive des variables ne signifie pas nécessairement une baisse de la qualité prédictive. Pour évaluer les performances réelles du modèle, il faut également disposer des résultats observés, par exemple les remboursements effectifs, puis suivre des métriques adaptées.

Les alertes de surveillance doivent être examinées avant toute décision de réentraînement ou de modification du modèle.

Les données et les rapports de surveillance ne doivent pas être considérés comme une preuve de conformité réglementaire ou comme une garantie de performance du modèle.

11. Limites et améliorations possibles

Les évolutions possibles comprennent :

Le déploiement sur une plateforme d'hébergement permanente.
Le renforcement de la validation des données d'entrée.
L'amélioration du suivi des performances du modèle.
L'amélioration de la surveillance par des alertes automatiques et le suivi des performances à partir des résultats réels.
12. Avertissement

Ce projet est réalisé dans un contexte pédagogique. Les prédictions ne doivent pas être utilisées seules pour prendre une décision réelle d'octroi de crédit.