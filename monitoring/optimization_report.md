Rapport d'optimisation du modèle — Projet Prêt à Dépenser



1\. Objectif



L'objectif de cette étape est d'évaluer les performances de l'API de scoring et de vérifier qu'une modification du processus d'inférence permet de réduire le temps de calcul sans modifier les résultats du modèle.



2\. Méthode utilisée



Le profilage du modèle a été réalisé à l'aide de cProfile, time.perf\_counter et psutil.



Une comparaison a également été effectuée entre deux versions de l'API exécutées dans Docker :



Version 1.4 : utilisation du pipeline complet avec model.predict\_proba(data).



Version 1.5 : utilisation directe du préprocesseur et du classifieur du pipeline.



Les mesures comparatives ont été réalisées sur le même payload de test, avec 5 requêtes de chauffe puis 20 requêtes mesurées par version.



3\. Résultats des mesures



Indicateur



Version 1.4



Version 1.5



Temps moyen



5,304 ms



4,980 ms



Temps médian



4,800 ms



4,646 ms



L'évolution du temps moyen représente une diminution mesurée d'environ 6,1 %.



L'évolution de la médiane représente une diminution d'environ 3,2 %.



Ces résultats indiquent une amélioration modeste des temps d'inférence dans les conditions du test.



4\. Vérification de la cohérence des prédictions



Le même dossier de test a été envoyé aux deux versions de l'API.



Score v1.4 : 0,903006



Score v1.5 : 0,903006



Écart : 0



Les deux versions produisent donc le même score pour cet exemple.



5\. Tests et compatibilité



Les vérifications réalisées ont donné les résultats suivants :



Tests automatisés : 8 tests réussis sur 8.



API v1.5 : démarrage et fonctionnement vérifiés dans Docker.



Endpoint /health : statut healthy.



Modèle chargé correctement.



Nombre de variables attendues : 304.



6\. Limites de l'expérience



Les résultats doivent être interprétés avec prudence :



Les mesures reposent sur un seul payload de test.



Chaque version a été évaluée sur 20 requêtes mesurées.



Les tests ont été réalisés localement dans Docker et non sur une infrastructure de production.



Les variations de charge de la machine peuvent influencer les temps mesurés.



L'égalité des scores sur un exemple ne démontre pas à elle seule l'absence de régression globale de précision ou de biais.



Aucun gain de consommation CPU ou mémoire n'est établi par cette comparaison.



Une validation complémentaire sur un jeu de données plus important serait nécessaire pour confirmer ces résultats.



7\. Conclusion



La version 1.5 conserve le même résultat de prédiction que la version 1.4 sur le dossier testé.



Les mesures montrent une réduction modeste du temps moyen et du temps médian d'inférence.



La version optimisée a été intégrée à l'API et son fonctionnement dans Docker a été vérifié.



Cette optimisation constitue une première amélioration technique. Des tests complémentaires sur un volume plus important de données et dans des conditions proches de la production restent nécessaires pour confirmer le gain de performance

