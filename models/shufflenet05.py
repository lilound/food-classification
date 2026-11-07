"""ShuffleNetV2 ×0,5 adapté aux catégories Food-11."""

from torch import nn
from torchvision.models import ShuffleNet_V2_X0_5_Weights, shufflenet_v2_x0_5


def load_model(num_classes, *, pretrained=True):
    weights = ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1 if pretrained else None
    model = shufflenet_v2_x0_5(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
