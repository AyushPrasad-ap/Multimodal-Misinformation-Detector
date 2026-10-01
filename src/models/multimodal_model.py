from __future__ import annotations

import torch
from torch import nn


class MultimodalClassifier(nn.Module):
    """Fuse frozen text/image encoders and classify samples."""

    def __init__(
        self,
        text_encoder: nn.Module,
        image_encoder: nn.Module,
        hidden_dim: int,
        dropout: float,
        num_classes: int,
        freeze_backbones: bool = True,
    ) -> None:
        super().__init__()

        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.freeze_backbones = freeze_backbones

        text_dim = text_encoder.embedding_dim
        image_dim = image_encoder.embedding_dim

        self.text_dim = text_dim
        self.image_dim = image_dim
        self.fusion_dim = text_dim + image_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        if self.freeze_backbones:
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad = False

            for parameter in self.image_encoder.parameters():
                parameter.requires_grad = False

            self.text_encoder.eval()
            self.image_encoder.eval()

    def train(self, mode: bool = True) -> MultimodalClassifier:
        """Keep frozen backbones in eval mode during classifier training."""
        super().train(mode)

        if self.freeze_backbones:
            self.text_encoder.eval()
            self.image_encoder.eval()

        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Return classification logits."""

        if self.freeze_backbones:
            with torch.no_grad():
                text_features = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                image_features = self.image_encoder(images)
        else:
            text_features = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            image_features = self.image_encoder(images)

        fused_features = torch.cat(
            [text_features, image_features],
            dim=1,
        )

        logits = self.classifier(fused_features)

        return logits


# ARCHITECTURE
#
# Text
#  ↓
# DistilBERT ──→ 768-D ──┐
#                        │
#                        ├── Concatenate → 1280-D
#                        │
# Image                  │
#  ↓                     │
# ResNet-18 ──→ 512-D ──┘
#                              ↓
#                          Linear(1280, 256)
#                              ↓
#                            ReLU
#                              ↓
#                          Dropout(0.3)
#                              ↓
#                          Linear(256, 2)
#                              ↓
#                           Logits
