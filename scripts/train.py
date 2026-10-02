from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets import (
    MultimodalCollator,
    MultimodalDataset,
)
from src.models import (
    ImageEncoder,
    MultimodalClassifier,
    TextEncoder,
)
from src.preprocessing import (
    ImageProcessor,
    TextProcessor,
)
from src.training import Trainer
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.logging_config import setup_logging
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"


def main() -> None:
    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    config = load_config(CONFIG_PATH)

    set_seed(config["project"]["seed"])

    logger = setup_logging(config["paths"]["log_dir"])

    device = get_device(config["training"]["device"])

    processed_dir = PROJECT_ROOT / config["paths"]["processed_dir"]

    checkpoint_dir = PROJECT_ROOT / config["paths"]["checkpoint_dir"]

    results_dir = PROJECT_ROOT / config["paths"]["results_dir"]

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Processors
    # ---------------------------------------------------------

    text_config = config["models"]["text"]

    text_processor = TextProcessor(
        model_name=text_config["name"],
        max_length=text_config["max_length"],
    )

    image_processor = ImageProcessor()

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------

    train_df = pd.read_csv(processed_dir / "train.csv")

    validate_df = pd.read_csv(processed_dir / "validate.csv")

    train_dataset = MultimodalDataset(
        dataframe=train_df,
        processed_dir=processed_dir,
        image_processor=image_processor,
    )

    validate_dataset = MultimodalDataset(
        dataframe=validate_df,
        processed_dir=processed_dir,
        image_processor=image_processor,
    )

    collator = MultimodalCollator(text_processor=text_processor)

    training_config = config["training"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["num_workers"],
        pin_memory=(training_config["pin_memory"] and device.type == "cuda"),
        collate_fn=collator,
    )

    validate_loader = DataLoader(
        validate_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config["num_workers"],
        pin_memory=(training_config["pin_memory"] and device.type == "cuda"),
        collate_fn=collator,
    )

    logger.info(
        "Train samples: %d",
        len(train_dataset),
    )

    logger.info(
        "Validation samples: %d",
        len(validate_dataset),
    )

    # ---------------------------------------------------------
    # Encoders
    # ---------------------------------------------------------

    text_encoder = TextEncoder(
        model_name=text_config["name"],
        freeze=text_config["freeze"],
    ).to(device)

    image_encoder = ImageEncoder(freeze=config["models"]["image"]["freeze"]).to(device)

    # ---------------------------------------------------------
    # Multimodal model
    # ---------------------------------------------------------

    fusion_config = config["models"]["fusion"]

    model = MultimodalClassifier(
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        hidden_dim=fusion_config["hidden_dim"],
        dropout=fusion_config["dropout"],
        num_classes=config["project"]["num_classes"],
        freeze_backbones=True,
    ).to(device)

    # ---------------------------------------------------------
    # Optimizer + loss
    # ---------------------------------------------------------

    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    criterion = nn.CrossEntropyLoss()

    logger.info(
        "Trainable parameters: %s",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    )

    # ---------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=validate_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=training_config["epochs"],
        checkpoint_dir=checkpoint_dir,
        mixed_precision=training_config["mixed_precision"],
        gradient_clip_norm=training_config["gradient_clip_norm"],
        early_stopping_patience=training_config["early_stopping_patience"],
        logger=logger,
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    history = trainer.fit()

    history_path = results_dir / "training_history.csv"

    history.to_csv(
        history_path,
        index=False,
    )

    logger.info(
        "Training history saved to %s",
        history_path,
    )


if __name__ == "__main__":
    main()
