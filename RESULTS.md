# Résultats

## ResNet18

Paramètres : 11 182 155

ResNet18, transfer learning, Food-11

| Metric | Value |
| --- | ---: |
| Frugality | 0.6087 |
| F1 | 0.8841 |
| Precision | 0.8489 |
| Accuracy | 0.9064 |
| Recall | 0.9223 |

## MobileNetV3-Small

Paramètres : 1 529 131

MobileNetV3-Small, ImageNet pretrained, fine-tuning, 224x224, random horizontal flip, Adam lr=1e-4, 5 epochs

| Metric | Value |
| --- | ---: |
| Frugality | 0.6726 |
| F1 | 0.8544 |
| Precision | 0.8284 |
| Accuracy | 0.8889 |
| Recall | 0.8822 |

## Comparaison et informations manquantes

- Sur ces résultats, MobileNetV3-Small a une frugalité supérieure de 0,0639 ; ResNet18 a un F1 supérieur de 0,0297.
- La date, la séparation des données, la graine, le matériel, la durée et les conditions complètes de chaque essai ne sont pas documentés. L’origine des scores et la méthode de calcul de la précision et du rappel ne sont pas précisées. Il faut les vérifier avant de conclure que les essais sont directement comparables.
- Le code actuel calcule le F1 macro et la frugalité comme `F1 − 0,3 / (1 + 1 / paramètres en millions)` ; il ne calcule actuellement ni précision ni rappel. Il reste à confirmer que les scores ci-dessus proviennent de ce calcul.

## Benchmark des taux d'apprentissage

Les dix modèles ont été comparés sur le même découpage local Food-11 :
10 637 images d'entraînement et 2 659 de validation, en 128 × 128, avec
les poids ImageNet et Adam. Trois taux ont été testés : fixe à `0,0001`,
adaptatif par lot et décroissance cosinus. Chaque modèle s'arrête
individuellement après stagnation, au plus tard à 60 époques.

Le taux **adaptatif** obtient la meilleure frugalité pour **9 modèles sur 10**.
L'exception est ShuffleNetV2 ×0,5, meilleur avec le taux fixe. En moyenne
sur les dix modèles, les meilleures frugalités sont de **0,6635** en adaptatif,
0,6555 en fixe et 0,6534 en cosinus.

### Top 5 par frugalité de validation

Pour chaque modèle, on retient le meilleur résultat parmi les trois taux.
L'époque indiquée est celle du meilleur score, pas nécessairement la dernière.

| Rang | Modèle | Taux privilégié | Frugalité | F1 macro | Paramètres | Meilleure époque |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | ShuffleNetV2 ×0,5 | Fixe | **0,6965** | 0,7748 | 353 067 | 22 |
| 2 | ShuffleNetV2 ×1,0 | Adaptatif | 0,6903 | 0,8578 | 1 264 879 | 8 |
| 3 | MobileNetV2 | Adaptatif | 0,6822 | 0,8896 | 2 237 963 | 17 |
| 4 | SqueezeNet 1.1 | Adaptatif | 0,6764 | 0,8029 | 728 139 | 10 |
| 5 | MNASNet 0.5 | Adaptatif | 0,6751 | 0,8213 | 951 603 | 43 |

Le nombre d'époques optimal dépend du modèle et du taux ; il sera étudié
plus précisément avec les trois premiers modèles :
[ShuffleNetV2 ×0,5](models/shufflenet05.py),
[ShuffleNetV2 ×1,0](models/shufflenet10.py) et
[MobileNetV2](models/mobilenet_v2.py). Ces résultats proviennent d'une seule
comparaison en validation locale ; ils ne sont pas des scores du test officiel.
Les graphiques et historiques sont dans `benchmark/modeles/resultats/nuit/`.

## Gel des couches des trois modèles retenus

Mêmes données de validation et protocole que le benchmark ci-dessus. Chaque
essai part des poids ImageNet ; le taux est fixe pour ShuffleNet ×0,5 et
adaptatif pour les deux autres. Le premier gel bloque `conv1` et `stage2`
sur ShuffleNet et `features[0:7]` sur MobileNetV2 ; le gel renforcé ajoute
`stage3` ou étend le gel à `features[0:14]`. Le gel maximal de ShuffleNet
×0,5 bloque aussi `stage4`. Les statistiques BatchNorm gelées restent fixes.

| Modèle | Gel | Meilleur F1 macro | Meilleure frugalité | Époque |
| --- | --- | ---: | ---: | ---: |
| ShuffleNet ×0,5 | Aucun | 0,7748 | 0,6965 | 22 |
| ShuffleNet ×0,5 | Premier | 0,7707 | 0,6924 | 18 |
| ShuffleNet ×0,5 | Renforcé | 0,7781 | **0,6998** | 30 |
| ShuffleNet ×0,5 | Maximal | 0,7716 | 0,6933 | 43 |
| ShuffleNet ×1,0 | Aucun | 0,8578 | 0,6903 | 8 |
| ShuffleNet ×1,0 | Premier | 0,8530 | 0,6854 | 18 |
| ShuffleNet ×1,0 | Renforcé | 0,8351 | 0,6675 | 14 |
| MobileNetV2 | Aucun | 0,8896 | 0,6822 | 17 |
| MobileNetV2 | Premier | 0,8836 | 0,6763 | 11 |
| MobileNetV2 | Renforcé | 0,8748 | 0,6675 | 4 |

Le gel renforcé améliore la frugalité de ShuffleNet ×0,5 de 0,0033 ; les
autres gels testés diminuent le score de leur modèle. Le nombre total de
paramètres et sa pénalité ne changent pas avec le gel : la variation du score
vient du F1. Chaque configuration n'a été exécutée qu'une fois, sur la
validation locale ; le léger gain reste à confirmer par répétition.
