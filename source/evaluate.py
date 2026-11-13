"""Export des prédictions au format JSON du challenge Food-11."""

import json
from pathlib import Path

import torch
from PIL import Image


def predict_test(model, test_dir, transform, classes, device, output_path, batch_size=32):
    """Prédit les classes des images de test et génère un JSON."""

    test_dir = Path(test_dir)
    output_path = Path(output_path)

    predictions = {}

    # Récupérer les images
    images = sorted(
        p for p in test_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    )

    if not images:
        raise FileNotFoundError(f"Aucune image trouvée dans {test_dir}")

    model.eval()

    with torch.no_grad():

        for start in range(0, len(images), batch_size):
            paths = images[start:start + batch_size]
            batch = []
            for image_path in paths:
                with Image.open(image_path) as image:
                    batch.append(transform(image.convert("RGB")))

            output = model(torch.stack(batch).to(device))
            for image_path, predicted_class in zip(paths, output.argmax(dim=1).tolist()):
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
