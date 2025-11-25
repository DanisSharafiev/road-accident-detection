import torch
import torch.nn as nn
from torchvision import models


class EfficientNetBaseline(nn.Module):
    """
    EfficientNet-B0 baseline for binary classification(road accident).
    Uses pretrained ImageNet weights.
    """
    def __init__(self, num_classes=2, pretrained=True, freeze_features=True, model_variant='b0'):
        super(EfficientNetBaseline, self).__init__()

        # Загружаем предобученную модель
        if model_variant == 'b0':
            if pretrained:
                self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                self.model = models.efficientnet_b0(weights=None)
        elif model_variant == 'b1':
            if pretrained:
                self.model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
            else:
                self.model = models.efficientnet_b1(weights=None)
        elif model_variant == 'b2':
            if pretrained:
                self.model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
            else:
                self.model = models.efficientnet_b2(weights=None)

        # Замораживаем feature extractor если нужно
        if freeze_features:
            for param in self.model.features.parameters():
                param.requires_grad = False

        # Получаем размер входа классификатора
        in_features = self.model.classifier[1].in_features

        # Заменяем классификатор для нашего числа классов
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
