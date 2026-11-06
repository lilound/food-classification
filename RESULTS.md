# Résultats

## ResNet18

Paramètres : 11 182 155

ResNet18, transfer learning, Food-11

| Metric | Value |
| --- | ---: |
| Frugality | 0.6087 |
| F1 | 0.8841 |
| Precision | 0.8489 |
| Accuracy | 0.9064 |
| Recall | 0.9223 |

## MobileNetV3-Small

Paramètres : 1 529 131

MobileNetV3-Small, ImageNet pretrained, fine-tuning, 224x224, random horizontal flip, Adam lr=1e-4, 5 epochs

| Metric | Value |
| --- | ---: |
| Frugality | 0.6726 |
| F1 | 0.8544 |
| Precision | 0.8284 |
| Accuracy | 0.8889 |
| Recall | 0.8822 |

## Comparaison et informations manquantes

- Sur ces résultats, MobileNetV3-Small a une frugalité supérieure de 0,0639 ; ResNet18 a un F1 supérieur de 0,0297.
- La date, la séparation des données, la graine, le matériel, la durée et les conditions complètes de chaque essai ne sont pas documentés. L’origine des scores et la méthode de calcul de la précision et du rappel ne sont pas précisées. Il faut les vérifier avant de conclure que les essais sont directement comparables.
- Le code actuel calcule le F1 macro et la frugalité comme `F1 − 0,3 / (1 + 1 / paramètres en millions)` ; il ne calcule actuellement ni précision ni rappel. Il reste à confirmer que les scores ci-dessus proviennent de ce calcul.
