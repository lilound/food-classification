# Définit et charge le réseau MobileNet.
# Adapte sa dernière couche pour classifier les 11 catégories alimentaires.

# Charge le réseau MobileNet préentraîné.
# Adapte la couche de sortie aux 11 catégories alimentaires.

import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def load_model(num_classes):

    model = mobilenet_v3_small(
        weights=MobileNet_V3_Small_Weights.DEFAULT
    )

    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        num_classes
    )

    return model