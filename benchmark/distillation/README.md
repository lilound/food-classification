# Distillation ResNet18 → ShuffleNetV2 ×0,5

Le programme entraîne ResNet18 sur le découpage Food-11 figé, puis entraîne un seul ShuffleNet par distillation. Les deux modèles utilisent des images de 128 × 128. Le taux de l'étudiant est adaptatif et ses couches `conv1`, `stage2` et `stage3` sont gelées, y compris les statistiques BatchNorm. La loss combine les vraies classes (75 %) et les sorties de l'enseignant (25 %, température 2).

Depuis la racine du projet :

```bash
.venv/bin/python -u benchmark/distillation/benchmark.py
```

Les résultats sont enregistrés dans `benchmark/distillation/execution/`. Chaque époque affiche le F1, la frugalité, le taux et le temps écoulé. Après une interruption, reprendre avec :

```bash
.venv/bin/python -u benchmark/distillation/benchmark.py --resume
```

Les checkpoints permettent de reprendre à la fin de la dernière époque complète. La reprise vérifie le protocole, le découpage et le code. Une nouvelle exécution indépendante exige un nouveau dossier avec `--output chemin/vers/dossier`.

À la fin, le meilleur checkpoint de chaque modèle, les historiques, le JSON de prédictions du test et les résultats de validation sont enregistrés. La section correspondante est ajoutée à `RESULTS.md` une seule fois. Les poids ImageNet doivent être disponibles en cache ou téléchargeables. Un entraînement sur CPU peut prendre plusieurs heures.
