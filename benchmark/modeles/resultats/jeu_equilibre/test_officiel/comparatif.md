# Food-11 équilibré — test officiel privé

Même jeu équilibré (1 600 images par classe), 128 × 128, Adam adaptatif et gel renforcé. Checkpoints choisis sur la validation. Le classement public n’a pas été modifié.

| Modèle | Époque testée | Arrêt | F1 officiel | Accuracy officielle | Frugalité officielle | Réponse |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| ShuffleNetV2 ×0,5 | 6 | Plateau, 12 époques | 0.7613 | 78.88% | 0.6830 | [Scores par classe](shufflenet05_e6_reponse.html) |
| ShuffleNetV2 ×1,0 | 3 | Interrompu après 6 époques | 0.8197 | 83.95% | 0.6521 | [Scores par classe](shufflenet10_e3_reponse.html) |
| MobileNetV2 | 3 | Interrompu après 5 époques | 0.8227 | 87.12% | 0.6154 | [Scores par classe](mobilenet_v2_e3_reponse.html) |
