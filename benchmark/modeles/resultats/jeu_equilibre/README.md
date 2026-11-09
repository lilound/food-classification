# Food-11 équilibré — résultats du benchmark

Jeu figé : 17 600 images d’entraînement (1 600 par classe), 2 659 images de validation et 3 097 images de test officiel. Entrées 128 × 128, graine 42, initialisation ImageNet, Adam et aucune augmentation aléatoire supplémentaire. [Description et empreinte du jeu](dataset.json).

[Courbes de validation](figures/frugalite_validation.svg) · [Historique de toutes les époques (CSV)](validation/historique.csv) · [Test officiel et détail par classe](test_officiel/comparatif.md)

| Essai | Meilleure époque | Dernière époque | Frugalité de validation | Arrêt | Historique |
| --- | ---: | ---: | ---: | --- | --- |
| ShuffleNetV2 ×0,5 · adaptatif + gel | 6 | 12 | 0.6905 | Plateau | [JSON](validation/shufflenet05_adaptive_strong/history.json) |
| ShuffleNetV2 ×1,0 · adaptatif + gel | 3 | 6 | 0.6649 | Interrompu | [JSON](validation/shufflenet10_adaptive_strong/history.json) |
| MobileNetV2 · adaptatif + gel | 3 | 5 | 0.6612 | Interrompu | [JSON](validation/mobilenet_v2_adaptive_strong/history.json) |
| ShuffleNetV2 ×0,5 · fixe sans gel | 11 | 11 | 0.6888 | Interrompu | [JSON](validation/shufflenet05_fixed/history.json) |

Le point sur chaque courbe marque le meilleur score de validation. Le gel renforcé fixe les blocs conv1, stage2 et stage3 de ShuffleNet ou features[0:14] de MobileNetV2, statistiques BatchNorm comprises.

Avec le réglage adaptatif initial, ShuffleNetV2 ×0,5 obtient la meilleure frugalité des trois checkpoints testés (**0,6830**). Les checkpoints ont été choisis en validation. Les deux autres essais ont été interrompus à des époques différentes.

Le réglage `loss-plateau` avec gel renforcé porte la frugalité officielle de ShuffleNetV2 ×0,5 à **0,6969** à l’époque 16. Le témoin sur le jeu original obtient **0,6743** à l’époque 8. [Comparaison, courbe et résultats bruts](comparaison_budget/README.md).
