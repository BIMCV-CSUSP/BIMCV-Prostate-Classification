from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.nets import EfficientNetBN


class EfficientNet_pretrained(nn.Module):
    def __init__(
        self,
        model_name: str = "efficientnet-b0",
        n_classes: int = 2,
        in_channels_eff: int = 1,
        pretrained_weights_path: str | None = None,
    ):
        super().__init__()

        model = EfficientNetBN(
            model_name=model_name,
            pretrained=False,
            progress=False,
            spatial_dims=3,
            in_channels=in_channels_eff,
            num_classes=n_classes,
        )

        if pretrained_weights_path:
            model.load_state_dict(torch.load(pretrained_weights_path)["state_dict"])

        self.model = model

    def forward(self, x):
        return self.model(x)


class EfficientNetMultimodal(nn.Module):
    def __init__(
        self,
        model_name: str = "efficientnet-b0",
        n_classes: int = 2,
        in_channels_eff: int = 1,
        in_num_features: int = 25,
        pretrained_weights_path: str | None = None,
    ):
        super().__init__()

        self.vision_backbone = EfficientNetBN(
            model_name=model_name,
            pretrained=True,
            progress=False,
            spatial_dims=3,
            in_channels=in_channels_eff,
            num_classes=n_classes,
        )

        if pretrained_weights_path:
            self.vision_backbone.load_state_dict(torch.load(pretrained_weights_path)["state_dict"])

        self.fc = nn.Linear(in_num_features + n_classes, n_classes)

    def forward(self, x_img, x_num):
        x = self.vision_backbone(x_img)
        x = torch.cat((x, x_num), dim=1)
        return self.fc(x)
