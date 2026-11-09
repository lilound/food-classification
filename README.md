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

### Générer le jeu équilibré à 1 600 images par classe

Avec le jeu officiel installé et Pillow disponible, exécutez depuis la racine :

```bash
python creer_dataset_equilibre.py
```

Le script crée `data/food-11-balanced/` : `train/` (17 600 images),
`validation/` (2 659) et `test/` (3 097). Il conserve chaque image
d'entraînement originale et complète les classes rares avec des variantes
déterministes (graine 42). Validation et test sont des copies à l'identique.
`manifest.json` relie chaque image à sa source et à sa recette ;
`dataset.json` enregistre les effectifs et l'empreinte SHA-256 du jeu.
Le script n'écrase aucun dossier existant. Pour un autre emplacement,
utilisez `--output` ; pour un autre quota, utilisez `--per-class` (au moins 1 600).

Transférez le dossier complet sur le deuxième PC et vérifiez-y l'empreinte :

```bash
python creer_dataset_equilibre.py --verify data/food-11-balanced
python benchmark/modeles/benchmark_lr.py --mode fixed --models shufflenet05 --data-dir data/food-11-balanced
```

Le benchmark utilise la validation fournie et mélange l'ordre du train sans
ajouter de retournements aléatoires. Pour lancer les trois modes sur ce jeu
dans des dossiers distincts du benchmark original :

```bash
python benchmark/modeles/lancer_benchmarks.py --data-dir data/food-11-balanced --runs benchmark/modeles/executions/equilibre --output benchmark/modeles/resultats/equilibre
```

Pour réexporter ces essais séparément, passez également
`--data-dir data/food-11-balanced` à `benchmark/modeles/export_benchmark.py`
avec leurs dossiers de résultats et un `--output` dédié.

Pour tester ShuffleNet ×0,5 avec taux adaptatif et gel renforcé
(`conv1`, `stage2`, `stage3`, BatchNorm comprises) :

```bash
python benchmark/modeles/benchmark_lr.py --mode adaptive --models shufflenet05 --freeze-early --freeze-level strong --data-dir data/food-11-balanced --output benchmark/modeles/executions/shufflenet05_adaptive_strong_balanced
```

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

### Variante expérimentale : taux adaptatif centré

```bash
python benchmark/modeles/benchmark_lr.py --mode adaptive --adaptive-policy centered --models shufflenet05 --freeze-early --freeze-level strong --data-dir data/food-11-balanced --output benchmark/modeles/executions/shufflenet05_centered_strong_balanced
```

Le contrôleur compare une moyenne rapide de l'alignement (coefficient 0,1) à
une moyenne lente (0,01). Leur écart pilote le taux autour d'une valeur de
base, sans cumuler les hausses quand le signal reste positif. Le taux varie
d'au plus un facteur 1,05 par lot ; en cas de plateau, la règle de validation
réduit le taux de base entre deux époques. L'historique enregistre les taux
minimum, moyen, maximum et la fraction de lots au plafond.

Sans `--adaptive-policy centered`, le comportement et la reprise de
l'adaptatif d'origine sont conservés. Comparez les deux variantes avec le
même découpage, le même gel et le même nombre de mises à jour.

Pour essayer la recherche de pas, utilisez `--adaptive-policy probe` et
un autre dossier de sortie. Tous les 20 lots, le contrôleur compare
`0,5 × LR`, `1 × LR` et `1,5 × LR` sur un échantillon du lot
d'entraînement suivant. Le journal affiche les pertes et le taux retenu.
Le checkpoint conserve le compteur des lots pour reprendre les contrôles
à la même cadence.

La variante `--adaptive-policy loss-plateau` fait monter le taux de
`1e-4` à `3e-4` pendant la première époque, puis le divise par deux
après deux époques sans baisse suffisante de la perte de validation
(seuil relatif de 0,2 %, plancher à `1e-5`). Elle garde toutes les mises
à jour d'Adam. Utilisez un dossier de sortie distinct ; la sélection des
poids et l'arrêt restent pilotés par la frugalité.

Pour comparer les résolutions, utilisez `--image-size 256` avec un dossier
de sortie distinct. La taille d'entrée s'applique à l'entraînement et à la
validation. Pour comparer à budget identique, fixez aussi le même
`--epoch-budget` pour les deux essais.

## Échéances et livrables

- **Date limite :** 15 novembre 2026.
- **Soutenance :** 16 novembre 2026.
- **Livrables :** le code sans les données et un rapport de 10 pages.

## Faire en vu du rendu

- Faire des graphiques benchmarks de différents modèles
- Faire graphiques différents facteur d'apprentissage


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
