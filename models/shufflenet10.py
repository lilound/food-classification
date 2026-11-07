"""ShuffleNetV2 ×1,0 adapté aux catégories Food-11."""

from torch import nn
from torchvision.models import ShuffleNet_V2_X1_0_Weights, shufflenet_v2_x1_0


def load_model(num_classes, *, pretrained=True):
    weights = ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
    model = shufflenet_v2_x1_0(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
