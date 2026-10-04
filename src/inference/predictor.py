from __future__ import annotations

import logging
from pathlib import Path

import torch

from src.models import (
    ImageEncoder,
    MultimodalClassifier,
    TextEncoder,
)
from src.preprocessing import (
    ImageProcessor,
    TextProcessor,
)


class Predictor:
    """Run inference on a single text + image sample."""

    def __init__(
        self,
        model: MultimodalClassifier,
        text_processor: TextProcessor,
        image_processor: ImageProcessor,
        device: torch.device,
        labels: dict[int, str],
        logger: logging.Logger | None = None,
    ) -> None:
        self.model = model
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.device = device
        self.labels = labels
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    @torch.inference_mode()
    def predict(
        self,
        text: str,
        image_path: str | Path,
    ) -> dict[str, float | int | str]:
        """Predict the class for one text + image pair."""

        if not text or not text.strip():
            raise ValueError("Text input must not be empty.")

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # -----------------------------------------------------
        # Text preprocessing
        # -----------------------------------------------------

        text_inputs = self.text_processor([text])

        input_ids = text_inputs["input_ids"].to(self.device)

        attention_mask = text_inputs["attention_mask"].to(self.device)

        # -----------------------------------------------------
        # Image preprocessing
        # -----------------------------------------------------

        image_tensor = self.image_processor(image_path)

        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # -----------------------------------------------------
        # Model inference
        # -----------------------------------------------------

        self.model.eval()

        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=image_tensor,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

        predicted_class = int(torch.argmax(probabilities).item())

        confidence = float(probabilities[predicted_class].item())

        label = self.labels[predicted_class]

        self.logger.info(
            "Inference completed | prediction=%s | confidence=%.4f",
            label,
            confidence,
        )

        return {
            "prediction": label,
            "class_id": predicted_class,
            "confidence": confidence,
            "probability_real": float(probabilities[0].item()),
            "probability_fake": float(probabilities[1].item()),
        }


def build_model(
    *,
    text_model_name: str,
    text_freeze: bool,
    image_freeze: bool,
    hidden_dim: int,
    dropout: float,
    num_classes: int,
    device: torch.device,
) -> MultimodalClassifier:
    """Build the exact multimodal architecture used during training."""

    text_encoder = TextEncoder(
        model_name=text_model_name,
        freeze=text_freeze,
    ).to(device)

    image_encoder = ImageEncoder(
        freeze=image_freeze,
    ).to(device)

    model = MultimodalClassifier(
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_classes=num_classes,
        freeze_backbones=True,
    ).to(device)

    return model
