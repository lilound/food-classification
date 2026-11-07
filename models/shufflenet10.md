# ShuffleNetV2 ×1,0

- Taux privilégié : **adaptatif par lot**, initialisé à 0,0001.
- Meilleure frugalité en validation : **0,6903**, à l'époque 8.
- F1 macro : 0,8578 ; accuracy : 85,22 % ; paramètres : 1 264 879.
- Fin de l'entraînement : époque 14, arrêt après stagnation.

Comparaison des meilleures frugalités : adaptatif 0,6903 ; fixe 0,6659 ;
cosinus 0,6654. Deuxième candidat pour approfondir le nombre d'époques et le taux.
