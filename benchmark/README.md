# Dossier du benchmark Food-11

Après export, ouvrir [index.md](index.md) pour accéder aux résultats
disponibles et voir quels essais sont encore en cours.

| Dossier ou fichier | Contenu |
| --- | --- |
| `experiences/<mode>/donnees_brutes/` | Métadonnées du run en JSON, historique de toutes les époques et classement par meilleur état en JSON et en CSV. |
| `courbes_par_modele/` | Une figure PNG et SVG par modèle, avec une courbe pour chaque mode testé sur ce modèle. |
| `comparaison/frugalite_toutes_courbes.{png,svg}` | Toutes les courbes dans un diagramme à axes partagés, avec un panneau par mode. |
| `comparaison/scores.csv` | Meilleur score, époque retenue, raison d'arrêt et score à la dernière époque commune aux modes disponibles pour chaque modèle. |
| `protocole/` | Copie du protocole figé et du découpage utilisés pour produire les figures. |

Le point sur une courbe indique le meilleur score. Les figures utilisent
la frugalité de validation et les époques effectivement mesurées. Les PNG
sont prêts à insérer ; les SVG conviennent à un rapport ou à un redimensionnement.
Le dossier se partage sans images Food-11 ni fichiers de poids. Les fichiers
générés sont locaux et ignorés par Git.

Depuis la racine du dépôt, pour réexporter les essais locaux :

```bash
python export_benchmark.py results/benchmark_lr_fixed_p6 results/benchmark_lr_variable_p6 results/benchmark_lr_adaptive
```

Lorsque l'autre PC renvoie ses résultats, copier ses fichiers `run.json`,
`history.json` et `summary.json` dans `results/benchmark_lr_variable_p6/`
ou passer leur dossier réel à la commande. Un dossier sans résultats est
signalé dans l'index et sera intégré dès que ses trois fichiers sont présents.
N'utiliser qu'un dossier par mode et le même protocole à patience 6.
Si l'actualisation automatique tourne sur ce PC, il suffit de copier les
fichiers : le dossier est régénéré au prochain passage.

Pour figer une version indépendante en vue du rendu, ajouter
`--output benchmark-rendu` et conserver l'intégralité de ce dossier.
