from __future__ import annotations

from pathlib import Path

from PIL import Image
from torchvision.models import ResNet18_Weights


class ImageProcessor:
    """Load and preprocess images for the pretrained ResNet-18 encoder."""

    def __init__(self) -> None:
        self.weights = ResNet18_Weights.DEFAULT
        self.transform = self.weights.transforms()

    def __call__(self, image_path: str | Path):
        """Load, convert to RGB, and preprocess one image."""
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return self.transform(image)
