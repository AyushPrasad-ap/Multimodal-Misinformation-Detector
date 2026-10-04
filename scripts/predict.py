from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.inference import Predictor, build_model
from src.preprocessing import (
    ImageProcessor,
    TextProcessor,
)
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.logging_config import setup_logging
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"


def parse_args() -> argparse.Namespace:
    """Parse command-line inference inputs."""

    parser = argparse.ArgumentParser(
        description="Run multimodal misinformation detection inference."
    )

    parser.add_argument(
        "--text",
        required=True,
        help="Text or headline to classify.",
    )

    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Path to the associated image.",
    )

    return parser.parse_args()


def main() -> None:
    # ---------------------------------------------------------
    # Arguments
    # ---------------------------------------------------------

    args = parse_args()

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    config = load_config(CONFIG_PATH)

    set_seed(config["project"]["seed"])

    logger = setup_logging(config["paths"]["log_dir"])

    device = get_device(config["training"]["device"])

    checkpoint_path = PROJECT_ROOT / config["paths"]["checkpoint_dir"] / "best_model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

    # ---------------------------------------------------------
    # Processors
    # ---------------------------------------------------------

    text_config = config["models"]["text"]
    image_config = config["models"]["image"]
    fusion_config = config["models"]["fusion"]

    text_processor = TextProcessor(
        model_name=text_config["name"],
        max_length=text_config["max_length"],
    )

    image_processor = ImageProcessor()

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = build_model(
        text_model_name=text_config["name"],
        text_freeze=text_config["freeze"],
        image_freeze=image_config["freeze"],
        hidden_dim=fusion_config["hidden_dim"],
        dropout=fusion_config["dropout"],
        num_classes=config["project"]["num_classes"],
        device=device,
    )

    # ---------------------------------------------------------
    # Load best checkpoint
    # ---------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    logger.info(
        "Loaded checkpoint: %s",
        checkpoint_path,
    )

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------

    labels = {
        int(class_id): label_name
        for class_id, label_name in config["project"]["labels"].items()
    }

    # ---------------------------------------------------------
    # Predictor
    # ---------------------------------------------------------

    predictor = Predictor(
        model=model,
        text_processor=text_processor,
        image_processor=image_processor,
        device=device,
        labels=labels,
        logger=logger,
    )

    result = predictor.predict(
        text=args.text,
        image_path=args.image,
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    print("\nMultimodal Misinformation Detector")
    print("=" * 42)

    print(f"Prediction     : {str(result['prediction']).upper()}")

    print(f"Confidence     : {float(result['confidence']) * 100:.2f}%")

    print(f"Probability Real: {float(result['probability_real']) * 100:.2f}%")

    print(f"Probability Fake: {float(result['probability_fake']) * 100:.2f}%")

    print("=" * 42)


if __name__ == "__main__":
    main()
