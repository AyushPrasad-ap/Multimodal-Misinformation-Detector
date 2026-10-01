from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    """Pretrained transformer text encoder."""

    def __init__(
        self,
        model_name: str,
        freeze: bool = True,
    ) -> None:
        super().__init__()

        self.model = AutoModel.from_pretrained(model_name)

        self.embedding_dim = self.model.config.hidden_size

        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

            self.model.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate a fixed-dimensional text representation.

        Returns:
            Tensor of shape [batch_size, hidden_size].
        """

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # DistilBERT uses the first token representation
        # ([CLS]) as the sequence-level representation.
        text_embedding = outputs.last_hidden_state[:, 0, :]

        return text_embedding
