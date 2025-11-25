import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViTBaseline(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, freeze_features=True):
        super(ViTBaseline, self).__init__()

        # 1. Загружаем ViT-B/16 с весами ImageNet
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = vit_b_16(weights=weights)

        # 2. Заморозка (Backbone)
        if freeze_features:
            for param in self.model.parameters():
                param.requires_grad = False

        # 3. Заменяем классификатор (Head)
        # У ViT-B hidden_dim = 768
        self.model.heads = nn.Linear(768, num_classes)

    def forward(self, x):
        # ---------------------------------------------------------
        # ФИКС: ViT требует строго 224x224.
        # Если пришло 512x512 (или другой размер), делаем ресайз.
        # ---------------------------------------------------------
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        return self.model(x)
