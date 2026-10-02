from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing import ImageProcessor


class MultimodalDataset(Dataset):
    """Dataset for paired text/image misinformation samples."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        processed_dir: str | Path,
        image_processor: ImageProcessor,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.processed_dir = Path(processed_dir)
        self.image_processor = image_processor

        required_columns = {
            "id",
            "text",
            "image_path",
            "label",
        }

        missing = required_columns - set(self.dataframe.columns)

        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataframe.iloc[index]

        image_path = self.processed_dir / row["image_path"]

        image = self.image_processor(image_path)

        return {
            "id": row["id"],
            "text": str(row["text"]),
            "image": image,
            "label": int(row["label"]),
        }


class MultimodalCollator:
    """Batch text tokenization and image tensors."""

    def __init__(
        self,
        text_processor,
    ) -> None:
        self.text_processor = text_processor

    def __call__(
        self,
        batch: list[dict[str, Any]],
    ) -> dict[str, Any]:

        texts = [item["text"] for item in batch]

        images = torch.stack([item["image"] for item in batch])

        labels = torch.tensor(
            [item["label"] for item in batch],
            dtype=torch.long,
        )

        ids = [item["id"] for item in batch]

        encoded = self.text_processor(texts)

        return {
            "ids": ids,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "images": images,
            "labels": labels,
        }
