# ShuffleNetV2 ×0,5 avec et sans équilibrage

Les deux essais utilisent 353 067 paramètres, des poids ImageNet, le gel de
`conv1`, `stage2` et `stage3`, des images de 128 × 128, des lots de 64,
Adam, le même warmup suivi de `loss-plateau`, la graine 42 et la même validation
de 2 659 images. La règle d'adaptation est commune et les taux dépendent de la
perte de validation de chaque essai.

Les deux essais valident toutes les 275 mises à jour. Le témoin conserve la
distribution des classes du jeu original. À chaque époque, il parcourt les
10 637 images, puis répète une partie du jeu choisie dans une seconde
permutation. Le jeu équilibré présente 17 600 images par époque, dont 6 963
variantes déterministes. Le témoin a été arrêté après 14 époques
(3 850 mises à jour), et l'essai équilibré après 22.

| Jeu d'entraînement | Meilleur checkpoint en validation | Frugalité validation | Frugalité test officiel | F1 macro test | Accuracy test |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 8 (2 200 mises à jour) | 0,6915 | 0,6743 | 0,7526 | 0,7924 |
| Équilibré | 16 (4 400 mises à jour) | 0,6930 | 0,6969 | 0,7752 | 0,7969 |

Sur la fenêtre commune des 14 premières époques, les meilleurs scores de
validation sont 0,6915 à l'époque 8 pour l'original et 0,6917 à l'époque 10 pour
l'équilibré. Sur le test officiel, les checkpoints ci-dessus ont été choisis
sur la validation et évalués en privé sur les mêmes 3 097 images. Le gain
observé de 0,0226 en frugalité compare deux recettes et deux durées
d'entraînement, avec une seule graine.

- [Courbe par mise à jour](frugalite_par_mise_a_jour.svg)
- [F1 par classe en validation pour les checkpoints sélectionnés](f1_par_classe_meilleurs_checkpoints.csv)
- [F1 par classe au test officiel](f1_par_classe_test_officiel.csv)
- [Réponse officielle pour le témoin](shufflenet05_original_e8_reponse.html)
- [Réponse officielle pour le jeu équilibré](../test_officiel/shufflenet05_loss_plateau_e16_reponse.html)
