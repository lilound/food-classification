# Évalue les performances du réseau de neurones.
# Calcule l'accuracy, le F1-score et le Frugality score.
# Génère également les prédictions sur les images de test.

import torch
import json

from pathlib import Path
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score


def evaluate(model, val_loader, device):
    """Évalue le modèle sur les images de validation."""

    model.eval()

    labels_real = []
    labels_pred = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            labels_real.extend(labels.tolist())
            labels_pred.extend(predictions.cpu().tolist())

    # Calcul des performances
    accuracy = accuracy_score(labels_real, labels_pred)

    f1 = f1_score(
        labels_real,
        labels_pred,
        average="macro",
        zero_division=0
    )

    # Nombre de paramètres (en millions)
    params = sum(p.numel() for p in model.parameters()) / 1_000_000

    # Frugality score
    frugality = f1 - 0.3 / (1 + 1 / params)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "frugality": frugality
    }


def predict_test(model, test_dir, transform, classes, device, output_path):
    """Prédit les classes des images de test et génère un JSON."""

    test_dir = Path(test_dir)
    output_path = Path(output_path)

    predictions = {}

    # Récupérer les images
    images = sorted(
        p for p in test_dir.rglob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    )

    if not images:
        raise FileNotFoundError(f"Aucune image trouvée dans {test_dir}")

    model.eval()

    with torch.no_grad():

        for image_path in images:

            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            output = model(image)
            predicted_class = output.argmax(dim=1).item()

            key = image_path.stem
            if key in predictions:
                raise ValueError(f"Nom d'image dupliqué : {key}")

            predictions[key] = classes[predicted_class].lower()

    # Sauvegarder le JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4)

    print(f"{len(predictions)} prédictions enregistrées dans {output_path}")

    return predictions