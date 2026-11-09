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

## ShuffleNetV2 ×0,5 sur Food-11 équilibré

[Historique, courbes et test officiel](benchmark/modeles/resultats/jeu_equilibre/README.md).

Les essais gardent les 2 659 images de validation du protocole initial.
L'entraînement utilise 17 600 images (1 600 par classe, dont 6 963 variantes
figées), sans augmentation aléatoire supplémentaire, en 128 × 128.

Un premier essai à taux fixe sans gel a été interrompu après 11 époques :
meilleure frugalité provisoire **0,6888**, F1 macro **0,7671** à l'époque 11.
L'essai retenu repart des poids ImageNet avec Adam adaptatif et le gel
renforcé de `conv1`, `stage2` et `stage3` repris de `origin/main` (`13be3fb`).
Ce protocole historique donnait 0,6998 avec taux fixe sur le jeu original ;
un écart avec le nouvel essai ne mesure pas à lui seul l'effet du rééquilibrage.

L'essai adaptatif avec gel renforcé a atteint une frugalité de **0,6905**
à l'époque **6** (F1 macro **0,7688**, accuracy **76,38 %**). Il s'est arrêté
sur plateau à l'époque **12**. Le taux était au plafond à l'époque 2 et
très proche aux époques 3 et 4 ; le plafond est passé de `3e-4` à `1,5e-4`
après l'époque 10. Sur le test officiel, l'évaluation privée du checkpoint de
l'époque 6 donne une frugalité de **0,6830** (F1 **0,7613**, accuracy **78,88 %**).

Sur le même jeu équilibré, MobileNetV2 a été entraîné à taux adaptatif
avec gel renforcé (`features[0:14]`). Essai interrompu après cinq époques.
Meilleur état de validation à l'époque 3 : **0,6612** de frugalité
(F1 macro **0,8686**). Son évaluation privée sur le test officiel donne
**0,6154** de frugalité, F1 **0,8227** et accuracy **87,12 %**.

ShuffleNetV2 ×1,0 a également été testé avec Adam adaptatif et gel renforcé
sur le jeu équilibré. Essai interrompu après six époques. Meilleur état de
validation à l'époque 3 : **0,6649** de frugalité (F1 macro **0,8325**).
Son évaluation privée donne **0,6521** de frugalité, F1 **0,8197** et
accuracy **83,95 %**. Parmi ces trois checkpoints évalués, ShuffleNetV2
×0,5 conserve la meilleure frugalité officielle (**0,6830**).
