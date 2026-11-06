## ResNet18

ResNet18, transfer learning, Food-11

| Metric | Value |
| --- | ---: |
| Frugality | 0.6087 |
| F1 | 0.8841 |
| Precision | 0.8489 |
| Accuracy | 0.9064 |
| Recall | 0.9223 |

## MobileNetV3-Small

MobileNetV3-Small, ImageNet pretrained, fine-tuning, 224x224, random horizontal flip, Adam lr=1e-4, 5 epochs

| Metric | Value |
| --- | ---: |
| Frugality | 0.6726 |
| F1 | 0.8544 |
| Precision | 0.8284 |
| Accuracy | 0.8889 |
| Recall | 0.8822 |

## Comparaison locale des modèles compacts (128 × 128)

Même découpage de Food-11 pour les cinq modèles (10 637 images pour
l'entraînement et 2 659 pour la validation, aucun doublon exact partagé).
Poids ImageNet, 5 époques, Adam à `1e-4`, lots de 64, retournement horizontal.
Les nombres de paramètres sont mesurés après adaptation aux 11 classes.

| Modèle | Paramètres | F1 macro | Frugalité |
| --- | ---: | ---: | ---: |
| SqueezeNet 1.1 | 728 139 | 0,7397 | **0,6133** |
| MobileNetV3 Small (référence) | 1 529 131 | 0,7836 | 0,6022 |
| ShuffleNetV2 ×1,0 | 1 264 879 | 0,7568 | 0,5892 |
| ShuffleNetV2 ×0,5 | 353 067 | 0,6219 | 0,5436 |
| MNASNet 0.5 | 951 603 | 0,4735 | 0,3273 |

À 5 époques, SqueezeNet gagne 0,0111 de frugalité sur MobileNet malgré un F1
inférieur de 0,0439, grâce à une réduction de 52 % du nombre de paramètres.
Ce sont des résultats de validation locale, distincts des scores du leaderboard
ci-dessus.

En prolongeant jusqu'à l'époque 7 sur le même découpage, ShuffleNetV2 ×1,0
passe en tête avec un F1 de 0,8191 et une frugalité de 0,6516. La première
reprise conserve les poids de l'époque 5 et réinitialise Adam ; l'état de
l'optimiseur est conservé pour la reprise suivante.

## Comparaison officielle des cinq candidats

Évaluation privée sur les 3 097 images du test, avec la mise à jour du
leaderboard décochée. Pour chaque modèle, le JSON utilise les meilleurs
poids de validation disponibles après la 19e époque de MNASNet.

| Modèle | Époque | Paramètres (M) | F1 officiel | Accuracy officielle | Frugalité officielle |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShuffleNetV2 ×0,5 | 16 | 0.353067 | 0,7677 | 0,7911 | **0,6895** |
| ShuffleNetV2 ×1,0 | 9 | 1.264879 | 0,8258 | 0,8544 | 0,6583 |
| SqueezeNet 1.1 | 11 | 0.728139 | 0,7738 | 0,8030 | 0,6474 |
| MobileNetV3 Small (nouvel essai à 128 × 128) | 10 | 1.529131 | 0,7783 | 0,8243 | 0,5969 |
| MNASNet 0.5 | 19 | 0.951603 | 0,7415 | 0,7656 | 0,5952 |

ShuffleNetV2 ×0,5 améliore de **0,0169** le score officiel de MobileNetV3
Small à 224 × 224 (0,6726) et utilise environ 77 % de paramètres en moins.
ShuffleNetV2 ×1,0 obtient le meilleur F1 des cinq essais compacts, mais sa
pénalité de taille le classe deuxième en frugalité.

Les prédictions envoyées, les réponses du serveur et la fiche de saisie
sont conservées localement dans `results/compact-128-official-e19/`.
