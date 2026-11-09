# Benchmark de distillation

- Enseignant : ResNet18 préentraîné ImageNet, réentraîné sur le découpage Food-11 figé en 128 × 128 ; retenir son meilleur F1 de validation.
- Étudiant : ShuffleNetV2 ×0,5 préentraîné, gel renforcé, taux adaptatif existant, 128 × 128 ; entraîner uniquement avec distillation (température 2, poids 0,25).
- Les deux modèles voient les mêmes images et retournements. La validation reste hors entraînement. F1 macro, frugalité, durée et checkpoints sont enregistrés à chaque époque.
- Reprendre après interruption sans mélanger les configurations ; exporter les prédictions du meilleur étudiant sur le test après l'entraînement. Reporter uniquement les résultats d'expériences terminées dans RESULTS.md.
