import torch
import torch.nn as nn
from torchvision import models


class ResNet50Baseline(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, freeze_features=True):
        super(ResNet50Baseline, self).__init__()

        # Загружаем предобученную модель ResNet50
        self.resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )

        # (опционально) замораживаем все слои кроме последнего FC
        if freeze_features:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Меняем последний fully connected layer под нашу задачу
        in_features = self.resnet.fc.in_features  # ResNet50: 2048
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
