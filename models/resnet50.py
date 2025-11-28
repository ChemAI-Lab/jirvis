import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet101_Weights, ResNet152_Weights


class jirvis(nn.Module):
    def __init__(self, num_labels, model_name="R50", model_type="standard", channels=3):
        super().__init__()
        self.model = ResNetFeatureExtractor(
            num_labels, model_name, model_type, channels
        )

    def forward(self, x):
        return self.model(x)


class ResidualMLPBlock(nn.Module):
    def __init__(self, features, dropout=0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(features, features),
            nn.BatchNorm1d(features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(features, features),
            nn.BatchNorm1d(features),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, num_labels, model_name, model_type, channels):
        super().__init__()

        # Get ResNet model
        resnet_map = {
            "R50": (models.resnet50, ResNet50_Weights.IMAGENET1K_V1),
            "R101": (models.resnet101, ResNet101_Weights.IMAGENET1K_V1),
            "R152": (models.resnet152, ResNet152_Weights.IMAGENET1K_V1),
        }
        model_fn, weights = resnet_map[model_name]
        resnet = model_fn(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.channels = channels

        # Get MLP based on type
        mlp_configs = {
            "standard": [(2048, 512, 0.4)],
            "3_layer": [(2048, 1024, 0.3), (1024, 512, 0.3)],
            "squeeze": [(2048, 512, 0.3), (512, 256, 0.3)],
            "4_layer": [(2048, 512, 0.3), (512, 256, 0.3), (256, 128, 0.3)],
        }

        if model_type == "residual":
            self.mlp = nn.Sequential(
                self._make_layer(2048, 512, 0.3),
                ResidualMLPBlock(512, 0.3),
                self._make_layer(512, 256, 0.3),
                nn.Linear(256, num_labels),
            )
        else:
            layers = []
            for in_feat, out_feat, dropout in mlp_configs[model_type]:
                layers.extend(self._make_layer(in_feat, out_feat, dropout))
            layers.append(nn.Linear(mlp_configs[model_type][-1][1], num_labels))
            self.mlp = nn.Sequential(*layers)

    def _make_layer(self, in_feat, out_feat, dropout):
        return [
            nn.Linear(in_feat, out_feat),
            nn.BatchNorm1d(out_feat),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]

    def forward(self, x):
        if self.channels == 1 and x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.mlp(x)
