Rapport d'optimisation du modèle — Projet Prêt à Dépenser



1\. Objectif



L'objectif de cette étape est d'évaluer les performances de l'API de scoring et de vérifier qu'une modification du processus d'inférence permet de réduire le temps de calcul sans modifier les résultats du modèle.



Cette étape vise également à observer la consommation des ressources système, notamment le processeur (CPU) et la mémoire vive (RAM), afin d'évaluer le comportement de l'API lors de sollicitations répétées.



2\. Méthode utilisée



Le profilage du modèle a été réalisé à l'aide des outils suivants :



cProfile : identification des fonctions les plus coûteuses en temps d'exécution.



time.perf\_counter : mesure précise des durées d'inférence.



psutil : observation des ressources système, notamment le CPU et la mémoire.



Docker : exécution et comparaison des différentes versions de l'API.



Une comparaison a été effectuée entre deux versions de l'API :



Version 1.4 : utilisation du pipeline complet avec la méthode model.predict\_proba(data).



Version 1.5 : utilisation directe des composants du pipeline, à savoir le préprocesseur (preprocessor) et le classifieur (classifier).



La version 1.5 applique directement les étapes de prétraitement et de classification, sans appeler la méthode de prédiction du pipeline complet.



Protocole de mesure des temps d'inférence



Les mesures comparatives ont été réalisées dans les conditions suivantes :



Même modèle de scoring.



Même dossier de test (payload).



5 requêtes de chauffe par version.



20 requêtes mesurées par version.



Exécution des deux versions dans des conteneurs Docker distincts.



Protocole de mesure CPU et mémoire



Une seconde expérience a été réalisée afin d'observer les ressources consommées sous charge.



Pour chaque version, 500 requêtes ont été envoyées à l'endpoint /predict, avec une exécution parallèle des deux traitements.



Les ressources des conteneurs ont été observées à l'aide de la commande docker stats --no-stream.



3\. Résultats des mesures de performance



3.1. Temps d'inférence



Les résultats obtenus lors de la comparaison des deux versions sont les suivants :



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



L'évolution du temps médian représente une diminution d'environ 3,2 %.



Ces résultats indiquent une amélioration modeste des temps d'inférence dans les conditions du test.



3.2. Consommation CPU et mémoire sous charge



Lors de l'expérience complémentaire, les mesures suivantes ont été relevées pendant l'exécution des 500 requêtes par version :



Indicateur



Version 1.4



Version 1.5



CPU



0,20 %



0,19 %



Mémoire utilisée



139,4 MiB



138,3 MiB



Mémoire relative



3,71 %



3,68 %



La mémoire utilisée est légèrement inférieure pour la version 1.5, avec un écart de 1,1 MiB, soit environ 0,8 %.



La différence de consommation CPU est de seulement 0,01 point. Elle est trop faible pour établir une amélioration significative.



Ces valeurs correspondent à un relevé instantané effectué pendant les tests. Elles ne représentent pas une moyenne de consommation sur toute la durée de l'expérience.



Les deux traitements de 500 requêtes se sont terminés sans erreur signalée par les tâches PowerShell.



4\. Vérification de la cohérence des prédictions



Afin de vérifier que l'optimisation ne modifie pas le comportement du modèle, le même dossier de test a été envoyé aux deux versions de l'API.



Les résultats obtenus sont les suivants :



Indicateur



Résultat



Score version 1.4



0,903006



Score version 1.5



0,903006



Écart entre les deux scores



0



Les deux versions produisent exactement le même score pour cet exemple.



Cette vérification confirme que la modification du processus d'inférence conserve le résultat de prédiction sur le dossier testé.



Elle ne constitue toutefois pas une validation complète de la qualité globale du modèle.



5\. Tests et compatibilité



Plusieurs vérifications ont été réalisées afin de contrôler le bon fonctionnement de la version optimisée.



Les résultats sont les suivants :



Tests automatisés : 8 tests réussis sur 8.



API v1.5 : démarrage et fonctionnement vérifiés dans Docker.



Endpoint /health : statut healthy.



Modèle chargé correctement.



Nombre de variables attendues : 304.



Tests de charge : 500 requêtes exécutées par version, sans erreur signalée par les tâches PowerShell.



Intégration continue (GitHub Actions) : workflow exécuté avec succès, comprenant les tests automatisés, la construction de l'image Docker et les vérifications de fonctionnement de l'API.



La version optimisée reste donc compatible avec l'environnement d'exécution testé.



6\. Limites de l'expérience



Les résultats obtenus doivent être interprétés avec prudence.



Plusieurs limites sont à prendre en compte :



Les mesures de latence reposent sur un seul payload de test.



Chaque version a été évaluée sur 20 requêtes mesurées pour la comparaison initiale.



Les tests ont été réalisés localement dans Docker et non sur une infrastructure de production.



Les variations de charge de la machine peuvent influencer les temps mesurés.



Les mesures CPU et mémoire sous charge reposent sur un relevé instantané et non sur une moyenne calculée sur toute la durée des tests.



L'exécution parallèle des deux versions peut influencer les ressources disponibles pour chaque conteneur.



L'égalité des scores sur un seul exemple ne démontre pas à elle seule l'absence de régression globale de précision ou de biais.



Aucun gain significatif de consommation CPU ou mémoire n'est établi par cette expérience.



Une validation complémentaire sur un jeu de données plus important serait nécessaire pour confirmer les résultats et vérifier la qualité globale du modèle.



7\. Conclusion



L'optimisation de l'API de scoring a permis d'obtenir une amélioration mesurée des temps d'inférence.



La comparaison entre les versions 1.4 et 1.5 montre une diminution d'environ 6,1 % du temps moyen et de 3,2 % du temps médian.



La cohérence des prédictions a également été vérifiée sur le dossier de test : les deux versions produisent le même score de 0,903006, sans différence.



Une expérience complémentaire de 500 requêtes par version a permis d'observer la consommation CPU et mémoire. La version 1.5 présente une consommation mémoire légèrement inférieure lors du relevé effectué, mais aucun gain significatif de ressources ne peut être confirmé à ce stade.



Enfin, la version optimisée a été intégrée à l'API, testée dans Docker et validée par le workflow CI/CD.



Cette optimisation constitue une première amélioration technique du processus d'inférence. Des tests complémentaires sur un volume plus important de données et dans des conditions proches de la production restent nécessaires pour confirmer durablement les gains de performance et vérifier l'absence de régression globale du modèle.

