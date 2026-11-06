"""Réseaux Torchvision préentraînés adaptés aux classes de Food-11."""

from torch import nn
from torchvision import models


BUILDERS = {
    "mobilenet": models.mobilenet_v3_small,
    "shufflenet05": models.shufflenet_v2_x0_5,
    "shufflenet10": models.shufflenet_v2_x1_0,
    "squeezenet": models.squeezenet1_1,
    "mnasnet05": models.mnasnet0_5,
}

LR_BUILDERS = {
    **BUILDERS,
    "mobilenet_v2": models.mobilenet_v2,
    "regnet_y_400mf": models.regnet_y_400mf,
    "efficientnet_b0": models.efficientnet_b0,
    "mobilenet_v3_large": models.mobilenet_v3_large,
    "efficientnet_b1": models.efficientnet_b1,
}


def load_model(name, num_classes, *, pretrained=True, weight_id="DEFAULT"):
    builder = LR_BUILDERS[name]
    weights = models.get_model_weights(builder)[weight_id] if pretrained else None
    model = builder(weights=weights)

    if name.startswith(("shufflenet", "regnet")):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "squeezenet":
        # SqueezeNet produit ses cartes de classes par une convolution 1 × 1.
        model.classifier[1] = nn.Conv2d(
            model.classifier[1].in_channels, num_classes, kernel_size=1
        )
        # Les cartes de scores restent signées, comme les logits des autres réseaux.
        model.classifier[2] = nn.Identity()
        model.num_classes = num_classes
    else:
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features, num_classes
        )

    return model
