# Food-11 · Code de rendu

Le modèle est un ShuffleNetV2 ×0,5 préentraîné sur ImageNet, adapté aux
11 classes Food-11. Les entrées sont redimensionnées en 512 × 512.
Les blocs `conv1`, `stage2` et `stage3` sont gelés, statistiques BatchNorm comprises.
Le modèle compte 353 067 paramètres, dont 299 883 entraînables.

## Reproduire l'entraînement

Placez le jeu Food-11 officiel dans `data/food-11/`, avec les images de
`train/` regroupées par classe et les images officielles dans `test/`.
Depuis la racine du dépôt, avec Python 3.14, exécutez les commandes suivantes.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m source.creer_dataset_equilibre
python -m source train
python -m source predict
```

Le fichier `source/protocole/split.json` fixe le découpage original. Il contient
les noms des fichiers répartis entre entraînement et validation. Le script
d'équilibrage conserve ce découpage et génère des variantes
déterministes à partir des images d'entraînement. Le jeu obtenu compte
1 600 images par classe à l'entraînement et 2 659 images en validation.
L'empreinte SHA-256 affichée pour le jeu utilisé dans le résultat de référence
est `fc027f4cd0659f961b1169d5a5b31ee87d8bc89c2057396664273baec59f3eff`.

L'entraînement tourne sur CPU, avec la graine 42, des lots de 64 et un
budget de 22 époques. Adam augmente le taux de `1e-4` à `3e-4` pendant
la première époque. Le taux est ensuite divisé par deux après deux
époques sans amélioration suffisante de la perte de validation
(seuil relatif de 0,2 %, plancher à `1e-5`).

Le fichier `results/history.json` contient les mesures par époque.
Le checkpoint `results/shufflenet05.pth` contient l'état de reprise et les
poids ayant obtenu la meilleure frugalité en validation. Utilisez
`python -m source train --resume` pour reprendre. La commande de prédiction
exporte les classes dans `results/predictions.json`. Le premier entraînement
peut télécharger les poids ImageNet.

## Résultat retenu

| Époque sélectionnée en validation | Frugalité de validation | Frugalité officielle | F1 macro officiel | Exactitude officielle |
| ---: | ---: | ---: | ---: | ---: |
| 15 | 0,8022 | 0,7885 | 0,8668 | 88,83 % |

Les scores officiels proviennent d'une évaluation privée sur les 3 097
images de test. Dans le formulaire officiel, saisissez **0,353067** pour le
nombre de paramètres en millions. [Testeur Food-11](https://www.creatis.insa-lyon.fr/kechichian/food11-classif-leaderboard.html).
