import torch
import torch.nn as nn
from torchvision import models


class InceptionV3Baseline(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, freeze_features=True):
        super(InceptionV3Baseline, self).__init__()

        # Загружаем предобученную модель InceptionV3
        # aux_logits=True обязателен для предобученных весов
        self.inception = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None,
            aux_logits=True  # Обязательно True для pretrained weights
        )

        # (опционально) замораживаем все слои кроме последнего FC
        if freeze_features:
            for param in self.inception.parameters():
                param.requires_grad = False

        # Меняем последний fully connected layer под нашу задачу
        in_features = self.inception.fc.in_features  # InceptionV3: 2048
        self.inception.fc = nn.Linear(in_features, num_classes)

        # Также меняем auxiliary classifier
        in_features_aux = self.inception.AuxLogits.fc.in_features
        self.inception.AuxLogits.fc = nn.Linear(in_features_aux, num_classes)

    def forward(self, x):
        return self.inception(x)
