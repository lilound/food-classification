# MobileNetV2

- Taux privilégié : **adaptatif par lot**, initialisé à 0,0001.
- Meilleure frugalité en validation : **0,6822**, à l'époque 17.
- F1 macro : 0,8896 ; accuracy : 88,49 % ; paramètres : 2 237 963.
- Fin de l'entraînement : époque 23, arrêt après stagnation.

Comparaison des meilleures frugalités : adaptatif 0,6822 ; fixe 0,6709 ;
cosinus 0,6714. Troisième candidat pour approfondir le nombre d'époques et le taux.
