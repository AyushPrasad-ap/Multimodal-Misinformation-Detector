from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ImageEncoder(nn.Module):
    """Pretrained ResNet-18 feature encoder."""

    def __init__(self, freeze: bool = True) -> None:
        super().__init__()

        weights = ResNet18_Weights.DEFAULT

        self.model = resnet18(weights=weights)

        # ResNet-18's classifier receives a 512-dimensional feature vector.
        self.embedding_dim = self.model.fc.in_features

        # Remove the ImageNet classification head.
        self.model.fc = nn.Identity()

        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

            self.model.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Generate image embeddings.

        Args:
            images: Tensor of shape [batch_size, 3, H, W].

        Returns:
            Tensor of shape [batch_size, 512].
        """
        return self.model(images)
