# Dossier du benchmark Food-11

## Lancer tous les essais et préparer l'archive

Depuis la racine du dépôt, installez les dépendances avec
`python -m pip install -r benchmark/modeles/protocole/requirements.txt` et placez le
dataset officiel dans `data/food-11/`. Puis lancez :

```bash
python benchmark/modeles/lancer_benchmarks.py
```

La commande lance les trois modes (`fixed`, `adaptive`, `cosine`), avec dix
modèles par mode. Chaque essai dispose de son dossier et sauvegarde ses
checkpoints dans `benchmark/modeles/executions/benchmark_nuit/`. Par défaut, un seul essai est
attribué à chaque GPU CUDA visible et les autres utilisent le CPU. Avec un
seul GPU, `--device cuda --jobs 1` permet d'enchaîner les trois sur le GPU.
Pour lancer seulement les deux méthodes variables, ajoutez
`--modes adaptive cosine`. Sur CPU, l'ensemble peut dépasser une nuit.

Pendant le calcul, `benchmark/modeles/resultats/nuit/` reçoit les historiques, les figures et
les journaux. Quand les essais se terminent, le lanceur crée
`benchmark/modeles/resultats/nuit.zip`, sans dataset ni fichiers de poids. En cas d'erreur
d'un essai, les autres continuent et `execution.json` indique lesquels sont
terminés. Un arrêt avec Ctrl+C produit aussi l'archive des époques déjà
sauvegardées. Pour reprendre, relancez la même commande avec `--resume`
et les mêmes modes : les essais déjà terminés sont conservés.

## Structure des résultats

Après export, ouvrir [index.md](index.md) pour accéder aux résultats
disponibles et voir quels essais sont encore en cours.

| Dossier ou fichier | Contenu |
| --- | --- |
| `experiences/<mode>/donnees_brutes/` | Copies des historiques et classements, recréées par l'export depuis `executions/benchmark_nuit/`. |
| `courbes_par_modele/` | Une figure PNG et SVG par modèle, avec une courbe pour chaque mode testé sur ce modèle. |
| `comparaison/frugalite_toutes_courbes.{png,svg}` | Toutes les courbes dans un diagramme à axes partagés, avec un panneau par mode. |
| `comparaison/scores.csv` | Meilleur score, époque retenue, raison d'arrêt et score à la dernière époque commune aux modes disponibles pour chaque modèle. |
| `protocole/` | Copie du protocole figé et du découpage utilisés pour produire les figures. |

Le point sur une courbe indique le meilleur score. Les figures utilisent
la frugalité de validation et les époques effectivement mesurées. Les PNG
sont prêts à insérer ; les SVG conviennent à un rapport ou à un redimensionnement.
Le dossier se partage sans images Food-11 ni fichiers de poids. Les fichiers
générés peuvent être partagés via Git ; les poids `.pth` restent ignorés.

Depuis la racine du dépôt, pour réexporter les essais locaux :

```bash
python benchmark/modeles/export_benchmark.py benchmark/modeles/executions/benchmark_lr_fixed_0.0001 benchmark/modeles/executions/benchmark_lr_adaptive benchmark/modeles/executions/benchmark_lr_cosine
```

Lorsque l'autre PC renvoie ses résultats, copier ses fichiers `run.json`,
`history.json` et `summary.json` dans `benchmark/modeles/executions/benchmark_lr_adaptive/`
ou passer leur dossier réel à la commande. Un dossier sans résultats est
signalé dans l'index et sera intégré dès que ses trois fichiers sont présents.
N'utiliser qu'un dossier par mode et le même protocole à patience 6. Le mode fixe doit utiliser un taux de 0,0001.
Si l'actualisation automatique tourne sur ce PC, il suffit de copier les
fichiers : le dossier est régénéré au prochain passage.

Pour figer une version indépendante en vue du rendu, ajouter
`--output benchmark-rendu` et conserver l'intégralité de ce dossier.
