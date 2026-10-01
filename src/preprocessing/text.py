from __future__ import annotations

from typing import Sequence

import torch
from transformers import AutoTokenizer


class TextProcessor:
    """Tokenize input text for a transformer encoder."""

    def __init__(
        self,
        model_name: str,
        max_length: int = 128,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(
        self,
        texts: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        """Tokenize a batch of texts."""
        return self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            return_tensors="pt",
        )
