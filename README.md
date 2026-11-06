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

- `MODEL_NAME` : modèle utilisé (`"mobilenet"` ou `"resnet"`) ;
- `EPOCHS` : nombre de passages sur les images d’entraînement ;
- `BATCH_SIZE` : nombre d’images traitées ensemble.

Les résultats sont créés dans `results/` :

- `<modèle>.pth` : paramètres appris du modèle ;
- `predictions_<modèle>.json` : prédictions sur le jeu de test, à déposer sur le site d’évaluation du challenge.

## Comparer les réseaux compacts

```bash
python benchmark_compact.py --epochs 5 --size 128 --output results/compact-128-seed42 --predict
```

Ce benchmark compare MobileNetV3 Small, ShuffleNetV2 ×0,5 et ×1,0, SqueezeNet 1.1
et MNASNet 0.5 avec des poids ImageNet et une tête adaptée aux 11 classes.
Il réentraîne MobileNet comme référence sur la même séparation et avec les mêmes
réglages. Les fichiers d'apprentissage identiques (SHA-256) restent dans le même
sous-ensemble. Le nombre de paramètres inclut la tête **Food-11**, et le score
de frugalité utilise ce total en millions.

Le dossier de sortie doit être nouveau. Il contient le découpage
(`split.json`), la configuration (`config.json`), les scores par époque
(`history.json`), le classement par meilleur score de validation
(`summary.json`) et les poids correspondants. L'option `--predict` génère
le JSON du meilleur candidat compact sur les images du test officiel.
Les scores mesurés par ce script sont des scores de validation locale.

Utilisez `--models mobilenet shufflenet05` pour limiter la comparaison à
deux réseaux, ou `--help` pour afficher les autres réglages.

Pour prolonger une comparaison terminée sans effacer ses résultats :

```bash
python benchmark_compact.py --resume-from results/compact-128-seed42 \
  --epochs 7 --size 128 --output results/compact-128-seed42-e7 --predict
```

`--epochs 7` signifie sept époques au total. Le script recharge les poids et
l'état d'Adam depuis les fichiers `last_<modèle>.pth`. Une ancienne expérience
sans ces fichiers peut être reprise si ses meilleurs poids correspondent à la
dernière époque de chaque modèle ; Adam est alors réinitialisé.
Avec `--patience 3 --min-delta 0.002`, l'entraînement s'arrête si aucun
modèle n'améliore d'au moins 0,002 son meilleur score de frugalité pendant
trois tours complets.

`--stop-on-drop` est une autre option : chaque modèle s'arrête dès qu'il
n'établit plus de record de frugalité. Les autres continuent, et les meilleurs
poids des modèles arrêtés restent disponibles pour l'export.

Les mesures de cette comparaison sont dans [RESULTS.md](RESULTS.md).

Pour préparer les trois meilleurs candidats au formulaire officiel :

```bash
python export_compact.py results/compact-128-seed42-e7 --top 3
```

Le sous-dossier `submission/` contient les JSON et la fiche
`formulaire.md`, avec le nombre de paramètres en millions et les mots-clés
pour chaque modèle. La fiche distingue validation locale et test officiel.

## Comparer taux fixe et taux variable sur deux PC

Sur la branche `taux_apprentissage_variable`, les deux modes utilisent
`benchmark_lr.py` et le même protocole versionné dans `benchmarks/lr/`.
Installez les versions épinglées dans un environnement Python compatible
(Python 3.14 utilisé pour préparer et vérifier le benchmark). Utilisez la
distribution PyTorch CPU ou CUDA adaptée à votre PC, aux versions indiquées.

```bash
python -m pip install -r benchmarks/lr/requirements.txt
```

Placez la distribution officielle Food-11 dans `data/food-11/`, ou ajoutez
`--data-dir chemin/vers/food-11` à la commande. Le script vérifie les noms,
l'empreinte du contenu et l'absence de fichiers identiques de part et d'autre
du découpage figé. Torchvision télécharge les poids ImageNet au premier besoin.

Sur le premier PC (Adam à `1e-4` sans scheduler) :

```bash
python benchmark_lr.py --mode fixed
```

Sur le deuxième PC (Adam au même taux initial, avec `ReduceLROnPlateau`) :

```bash
python benchmark_lr.py --mode plateau
```

Les deux modes comparent les mêmes dix réseaux, de 353 067 à 6 527 275
paramètres. Chaque réseau s'arrête individuellement après six époques sans
progrès d'au moins 0,001, jamais avant l'époque 12, ou au plafond de 60 époques.
La meilleure frugalité brute est sauvegardée indépendamment de ce seuil. La
patience ne se remet pas à zéro lors d'un changement de taux.

Si un benchmark a déjà démarré avec l'ancienne patience de dix, arrêtez
l'ancien processus puis importez ses checkpoints dans un nouveau dossier :

```bash
python benchmark_lr.py --mode plateau --import-patience10 results/benchmark_lr_variable --output results/benchmark_lr_variable_p6
```

Pour le taux fixe, utilisez `--mode fixed` et remplacez `variable` par `fixed`
dans ces deux chemins. Le script rejoue le nouvel arrêt sur les scores déjà
mesurés et garde les anciens fichiers à part. Il ne réentraîne que les modèles
dont le meilleur état admissible n'est plus disponible. Après l'import, utilisez
`--resume --output <nouveau_dossier>` si le nouveau processus est interrompu.

Le script sélectionne CUDA si disponible, sinon le CPU. Sur CPU, la durée
totale peut atteindre plusieurs heures. En cas d'interruption, relancez la
même commande avec `--resume` pour reprendre à la dernière époque sauvegardée,
avec l'état d'Adam et celui du scheduler. N'exécutez pas deux commandes
simultanément dans le même dossier de résultats.

Les dossiers locaux `results/benchmark_lr_fixed/` et
`results/benchmark_lr_variable/` contiennent `run.json`, `history.json`,
`summary.json`, les poids et, en fin de run, le graphique du mode.
Pour comparer les deux modes, transférez les trois JSON de chaque PC, sans les
poids. Une fois les fichiers réunis, lancez :

```bash
python plot_benchmark_lr.py results/benchmark_lr_fixed results/benchmark_lr_variable --output results/comparaison_lr
```

Si vous avez importé un ancien benchmark, indiquez son dossier `_p6` dans
cette commande. Elle produit deux graphiques PNG et SVG sur des axes communs,
ainsi qu'un comparatif par modèle au meilleur score et à la dernière époque
commune. Les durées dépendent du PC et ne servent pas à attribuer un gain au
scheduler. Les scores sont des scores de validation locale.

### Essai optionnel : taux adapté à chaque lot

```bash
python benchmark_lr_adaptive.py --models shufflenet05
```

Ce troisième essai garde Adam et le protocole commun, mais ajuste le taux
global avant chaque mise à jour, à partir de l'alignement entre le gradient
de la tête de classification et le déplacement utile du lot précédent.
Le signal est lissé ; le taux part de `1e-4` et reste entre `1e-6` et
`3e-4`. La stagnation de la frugalité en validation abaisse ce plafond
avec la même règle `ReduceLROnPlateau` que l'autre essai variable.

Les résultats vont dans `results/benchmark_lr_adaptive/`. L'historique
enregistre aussi le taux minimum, maximum et moyen de chaque époque.
Utilisez `--resume` avec la même sélection après une interruption, ou
omettez `--models` dans un nouveau dossier pour essayer les dix réseaux.
Comparez d'abord ShuffleNet ×0,5 aux deux autres modes sur le même découpage,
au meilleur score et à nombre d'époques égal.

Le [dossier benchmark](benchmark/README.md) rassemble les historiques JSON/CSV,
les courbes par modèle et le diagramme d'ensemble pour le rendu. Le guide
indique comment y intégrer les résultats du deuxième PC.

Pour lancer les trois modes en une commande et préparer l'archive
`benchmark/nuit.zip`, utilisez `python lancer_benchmarks.py`.

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
