# Food-11 — classification d’images

Projet de classification d’images alimentaires avec des réseaux de neurones. Le modèle apprend à reconnaître les 11 catégories du jeu de données Food-11.

## Consignes et documents

- [Consignes du projet sur Moodle](https://moodle.insa-lyon.fr/pluginfile.php/463690/mod_resource/content/3/food-11.html)
- [Compte rendu](https://1drv.ms/w/c/1265599f2c973b15/IQDkAhwM534kSIbf0eIzCYRUAQXxQlNQrV8-rmVndYI0OIc?e=mKdDRQ)
- [Comparaison de modèles PyTorch](https://www.wirelog.net/posts/2025-05-11-image-classification-models-pytorch-comparision/)

## Données attendues

Télécharger et extraire le jeu de données dans `data/food-11/`, avec les dossiers `train/` et `test/` :

```text
data/food-11/
├── train/
│   ├── bread/
│   ├── dairy/
│   ├── dessert/
│   └── ...
└── test/
    ├── 0.jpg
    ├── 1.jpg
    └── ...
```

Chaque sous-dossier de `train/` correspond à une catégorie alimentaire. Les images de `test/` sont directement dans ce dossier et ne sont pas classées par catégorie.

Les données ne sont pas incluses dans Git. Il faut donc les télécharger séparément avant de lancer le programme.

## Installation

Depuis le dossier du projet, créer puis activer un environnement Python :

```bash
python -m venv .venv
```

Sous Linux ou macOS :

```bash
source .venv/bin/activate
```

Sous Windows PowerShell :

```powershell
.venv\Scripts\Activate.ps1
```

Installer les bibliothèques nécessaires :

```bash
pip install -r requirements.txt
```

## Lancer le programme

Vérifier que les données sont dans `data/food-11/`, puis exécuter :

```bash
python main.py
```

Le programme entraîne le modèle, calcule ses scores sur une partie de `train/` réservée à la validation, sauvegarde le modèle et prédit les catégories des images de `test/`. Au premier lancement, torchvision peut télécharger les poids ImageNet pré-entraînés du modèle choisi.

Les principaux réglages se trouvent au début de `main.py` :

- `MODEL_NAME` : `"mobilenet"`, `"shufflenet05"`, `"shufflenet10"` ou `"mobilenet_v2"` ;
- `EPOCHS` : nombre de passages sur les images d’entraînement ;
- `BATCH_SIZE` : nombre d’images traitées ensemble.

Les trois modèles retenus par le benchmark ont chacun leur fichier dans
`models/` : `shufflenet05.py`, `shufflenet10.py` et `mobilenet_v2.py`.
Ils chargent les poids ImageNet et adaptent la sortie aux classes Food-11.
Leur taux privilégié est fixe pour ShuffleNet ×0,5 et adaptatif pour les deux
autres ; `main.py` utilise toutefois son propre entraînement à taux fixe.

Les résultats sont créés dans `results/` :

- `<modèle>.pth` : paramètres appris du modèle ;
- `predictions_<modèle>.json` : prédictions sur le jeu de test, à déposer sur le site d’évaluation du challenge.

## Comparer les modèles : taux fixe, adaptatif et cosinus

Les trois modes utilisent `benchmark/modeles/benchmark_lr.py` et le découpage figé dans
`benchmark/modeles/protocole/`. Installez les dépendances et placez Food-11 dans
`data/food-11/` (ou utilisez `--data-dir`). Les poids ImageNet sont téléchargés
au premier besoin.

```bash
python -m pip install -r benchmark/modeles/protocole/requirements.txt
python benchmark/modeles/benchmark_lr.py --mode fixed
python benchmark/modeles/benchmark_lr.py --mode adaptive
python benchmark/modeles/benchmark_lr.py --mode cosine
```

Le mode fixe utilise Adam avec un taux de **0,0001**. Le mode adaptatif
commence à **0,0001** et ajuste le taux avant chaque lot selon l'alignement
entre le gradient et le déplacement précédent. Il conserve son plafond,
baissé lorsque la frugalité de validation stagne. Le mode cosinus réduit le taux
à chaque époque de 0,0001 vers 0,000001 sur 60 époques ; l'arrêt anticipé peut
intervenir avant. Les trois modes partent du même taux initial.

Les dix modèles sont entraînés un par un, jusqu'à 60 époques, avec un arrêt
individuel après six époques sans progrès suffisant et jamais avant 12 époques.
La sortie indique le F1, la frugalité et les taux minimum et maximum par époque.
Pour limiter les modèles, ajoutez par exemple `--models shufflenet05 mobilenet`.

Les résultats et checkpoints sont écrits dans `benchmark/modeles/executions/benchmark_lr_fixed_0.0001/`,
`benchmark/modeles/executions/benchmark_lr_adaptive/` ou `benchmark/modeles/executions/benchmark_lr_cosine/`. Pour reprendre après interruption, relancez
la même commande avec `--resume` et la même sélection de modèles. Les anciens
résultats restent intacts, mais un ancien taux fixe ou plateau ne peut pas être
repris avec le nouveau mode fixe. Ne lancez pas deux processus dans le même dossier.
Le script utilise CUDA si disponible, sinon le CPU ; un benchmark CPU peut durer
plusieurs heures.

Pour générer les graphiques et tableaux à partir des trois dossiers :

```bash
python benchmark/modeles/export_benchmark.py benchmark/modeles/executions/benchmark_lr_fixed_0.0001 benchmark/modeles/executions/benchmark_lr_adaptive benchmark/modeles/executions/benchmark_lr_cosine
```

Voir le [guide des résultats](benchmark/modeles/resultats/README.md) et les
conclusions dans [RESULTS.md](RESULTS.md).

Pour lancer les trois modes en une commande et préparer l'archive
`benchmark/modeles/resultats/nuit.zip`, utilisez `python benchmark/modeles/lancer_benchmarks.py`.

## Échéances et livrables

- **Date limite :** 15 novembre 2026.
- **Soutenance :** 16 novembre 2026.
- **Livrables :** le code sans les données et un rapport de 10 pages.

## Faire en vu du rendu

- Faire graphiques différents facteur d'apprentissage


## Pipeline d’amélioration

Comparer plusieurs modèles préentraînés. Gardez les mêmes données de validation et, autant que possible, les mêmes conditions d’entraînement : résolution, augmentations, époques et taux d’apprentissage. Comparez leur frugalité, leur F1, leur nombre de paramètres et leur temps d’entraînement. Gardez les 2 ou 3 meilleurs pour la suite.

Tester le taux d’apprentissage et le nombre d’époques. Ce sont les premiers réglages à essayer sur les modèles retenus : un modèle mal entraîné peut sembler moins bon qu’il ne l’est. Par exemple, testez quelques taux autour de 1e-4, puis observez le score de validation après chaque époque. Un taux qui diminue au cours de l’entraînement peut aussi être testé.

Tester le gel des couches. Comparez trois possibilités : entraîner seulement la dernière couche, dégeler les dernières couches, puis entraîner tout le modèle. Geler des couches peut réduire le coût d’entraînement, mais aussi réduire le F1. Cela ne diminue pas nécessairement le nombre total de paramètres du modèle. PyTorch : apprentissage par transfert

Tester la taille des images. Commencez par 224 × 224 pixels — pas 224 × 244 — puis comparez avec une résolution plus petite, par exemple 160 × 160. Les images plus petites réduisent généralement le calcul, mais peuvent faire perdre des détails utiles. Respectez aussi les prétraitements associés aux poids préentraînés.

Tester l’équilibrage des catégories. Regardez d’abord si certaines classes sont sous-représentées et si leur F1 est faible. Si oui, essayez un échantillonnage équilibré ou une pondération de la fonction de perte. Comparez avec l’entraînement sans équilibrage : ce n’est pas automatiquement meilleur.

Tester les augmentations d’images. Le retournement horizontal avec une probabilité de 50 % est une bonne base, suivi d’essais modérés sur la luminosité et le contraste. Ces transformations s’appliquent à l’entraînement, pas à la validation. Elles peuvent améliorer la généralisation, mais il faut mesurer leur effet. PyTorch : transformations d’images

Recomparer les finalistes. Les réglages optimaux peuvent différer selon l’architecture : entraînez les meilleurs candidats avec leurs bons réglages, puis choisissez celui qui obtient la meilleure frugalité sur le même jeu de validation. Réservez le test final pour l’évaluation finale.

## Pistes d’amélioration

- Équilibrer le nombre d’images d’entraînement par catégorie. Par exemple, après plusieurs époques, le modèle devrait avoir été entraîné sur autant d’images de pain que de riz.
- Geler les premières couches et ne réentraîner que les dernières.
- Remplacer le taux d’apprentissage fixe par un taux variable.
- Changer la luminosité, le contraste, les couleurs
- Tester d’autres modèles :

Les tailles ci-dessous sont celles des modèles avec leur tête ImageNet.
Pour le score du challenge, utilisez le nombre de paramètres après adaptation à Food-11.

| Modèle | Paramètres (M) | Pénalité Frugality | Priorité |
|--------|----------------|-------------------|----------|
| ShuffleNetV2 ×0.5 | 1.4 | 0.175 | Très haute |
| SqueezeNet 1.1 | 1.2 | 0.164 | Haute |
| ShuffleNetV2 ×1.0 | 2.3 | 0.209 | Très haute |
| MobileNetV3-Small | 2.5 | 0.214 | Très haute |
| MNASNet 0.5 | 2.2 | 0.206 | Moyenne |
| MobileNetV2 | 3.5 | 0.233 | Haute |
| RegNetY-400MF | 4.3 | 0.243 | Haute |
| EfficientNet-B0 | 5.3 | 0.252 | Très haute |
| MobileNetV3-Large | 5.5 | 0.254 | Très haute |
| EfficientNet-B1 | 7.8 | 0.266 | Haute |
| EfficientNet-B2 | 9.1 | 0.270 | Moyenne |
| ResNet18 | 11.7 | 0.276 | Haute (référence) |

Bon courage à toute l’équipe et amusez-vous !
