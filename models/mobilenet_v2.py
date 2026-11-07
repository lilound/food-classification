"""MobileNetV2 adapté aux catégories Food-11."""

from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def load_model(num_classes, *, pretrained=True):
    weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v2(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model
