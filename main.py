# Point d'entrée du programme.
# Charge les données et le modèle, lance l'entraînement puis l'évaluation.

# Point d'entrée : chargement, entraînement, évaluation et prédictions.

from pathlib import Path
import torch

from models.mobilenet import load_model as load_mobilenet
from models.resnet import load_model as load_resnet

from src.dataset import find_data_folder, load_data, get_transforms
from src.train import train
from src.evaluate import evaluate, predict_test


# Configuration
DATA_DIR = Path("data/food-11")
MODEL_NAME = "mobilenet"  # "mobilenet" ou "resnet"
EPOCHS = 5
BATCH_SIZE = 32


def main():

    # 1. Charger les données
    train_dir = find_data_folder(DATA_DIR, ("train", "training"))

    train_loader, val_loader, classes = load_data(
        train_dir, BATCH_SIZE
    )

    # 2. Charger le modèle
    if MODEL_NAME == "mobilenet":
        model = load_mobilenet(len(classes))

    elif MODEL_NAME == "resnet":
        model = load_resnet(len(classes))

    else:
        raise ValueError("Modèle inconnu")

    # Choisir le GPU si disponible
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)

    # 3. Entraîner le modèle
    train(model, train_loader, device, epochs=EPOCHS)

    # 4. Évaluer le modèle
    scores = evaluate(model, val_loader, device)

    print("Accuracy :", scores["accuracy"])
    print("F1-score :", scores["f1"])
    print("Frugality :", scores["frugality"])

    # 5. Sauvegarder le modèle
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    torch.save(
        model.state_dict(),
        results_dir / f"{MODEL_NAME}.pth"
    )

    # 6. Générer les prédictions du test
    test_dir = find_data_folder(DATA_DIR, ("test", "testing"))

    predict_test(
        model,
        test_dir,
        get_transforms(train=False),
        classes,
        device,
        results_dir / f"predictions_{MODEL_NAME}.json"
    )


if __name__ == "__main__":
    main()