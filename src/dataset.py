# Chargement du dataset Food-11.
# Prépare les images et sépare les données en entraînement et validation.

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


def find_data_folder(root, possible_names):
    # Recherche un dossier train/training ou test/testing
    root = Path(root)

    for folder in root.iterdir():
        if folder.is_dir() and folder.name.lower() in possible_names:
            return folder

    raise FileNotFoundError(f"Dossier introuvable dans {root}")


def get_transforms(train=False):
    # Préparation des images pour le réseau de neurones

    operations = [transforms.Resize((224, 224))]

    if train:
        operations.append(transforms.RandomHorizontalFlip())

    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transforms.Compose(operations)


def load_data(train_dir, batch_size=32):
    # Charger les images
    train_data = datasets.ImageFolder(
        train_dir,
        transform=get_transforms(train=True)
    )

    val_data = datasets.ImageFolder(
        train_dir,
        transform=get_transforms(train=False)
    )

    # Séparer les données : 80 % entraînement / 20 % validation
    indices = list(range(len(train_data)))

    train_ids, val_ids = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=train_data.targets
    )

    # Créer les DataLoaders
    train_loader = DataLoader(
        Subset(train_data, train_ids),
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        Subset(val_data, val_ids),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, train_data.classes