"""Préparation des images pour ShuffleNetV2 initialisé sur ImageNet."""

from torchvision import transforms


def get_transforms(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
