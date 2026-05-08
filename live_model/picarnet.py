import torch.nn as nn
from torchvision import models


class PiCarNet(nn.Module):
    def __init__(self, pretrained=True, dropout_rate_first=0.3, dropout_rate_second=0.2):
        super().__init__()
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        in_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate_first),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate_second),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.head(self.backbone(x))