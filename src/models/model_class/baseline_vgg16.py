import torch
import torch.nn as nn
from torchvision import models

class VGG16Baseline(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, freeze_features=True):
        super(VGG16Baseline, self).__init__()

        # Загружаем предобученную модель VGG16
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)

        # (опционально) замораживаем свёрточные слои
        if freeze_features:
            for param in self.vgg.features.parameters():
                param.requires_grad = False

        # Меняем классификатор под нашу задачу
        in_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vgg(x)
