# DenseNet-201 with a 4-class head (dropout 0.3 + linear).
import torch.nn as nn
from torchvision import models


def build_model(num_classes=4, pretrained=True):
    weights = models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet201(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier.in_features, num_classes)
    )
    return model
