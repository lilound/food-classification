# Définit et charge le réseau ResNet.
# Adapte sa dernière couche pour classifier les 11 catégories alimentaires.

# Charge le réseau ResNet18 préentraîné.
# Adapte la couche de sortie aux 11 catégories alimentaires.

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def load_model(num_classes):

    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model