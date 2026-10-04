from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets import (
    MultimodalCollator,
    MultimodalDataset,
)
from src.evaluation import Evaluator
from src.models import (
    ImageEncoder,
    MultimodalClassifier,
    TextEncoder,
)
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


def main() -> None:
    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    config = load_config(CONFIG_PATH)

    set_seed(
        config["project"]["seed"],
    )

    logger = setup_logging(
        config["paths"]["log_dir"],
    )

    device = get_device(
        config["training"]["device"],
    )

    processed_dir = PROJECT_ROOT / config["paths"]["processed_dir"]

    checkpoint_dir = PROJECT_ROOT / config["paths"]["checkpoint_dir"]

    results_dir = PROJECT_ROOT / config["paths"]["results_dir"]

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Test data
    # ---------------------------------------------------------

    test_path = processed_dir / "test.csv"

    test_df = pd.read_csv(test_path)

    # Preserve the exact dataframe ordering so that prediction
    # outputs can be matched back to sample IDs.
    test_df = test_df.reset_index(drop=True)

    logger.info(
        "Test samples: %d",
        len(test_df),
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
    # Dataset + DataLoader
    # ---------------------------------------------------------

    test_dataset = MultimodalDataset(
        dataframe=test_df,
        processed_dir=processed_dir,
        image_processor=image_processor,
    )

    collator = MultimodalCollator(
        text_processor=text_processor,
    )

    training_config = config["training"]

    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config["num_workers"],
        pin_memory=(training_config["pin_memory"] and device.type == "cuda"),
        collate_fn=collator,
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    text_encoder = TextEncoder(
        model_name=text_config["name"],
        freeze=text_config["freeze"],
    ).to(device)

    image_encoder = ImageEncoder(
        freeze=config["models"]["image"]["freeze"],
    ).to(device)

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
    # Load best checkpoint
    # ---------------------------------------------------------

    checkpoint_path = checkpoint_dir / "best_model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

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
        "Loaded best checkpoint: %s",
        checkpoint_path,
    )

    logger.info(
        "Checkpoint epoch: %s",
        checkpoint.get("epoch", "unknown"),
    )

    logger.info(
        "Checkpoint validation loss: %s",
        checkpoint.get("val_loss", "unknown"),
    )

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        logger=logger,
    )

    (
        y_true,
        y_pred,
        probabilities,
    ) = evaluator.predict()

    (
        metrics,
        classification_report_data,
        confusion,
        false_positive_rate,
        true_positive_rate,
    ) = evaluator.compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
    )

    # ---------------------------------------------------------
    # Prediction artifact
    # ---------------------------------------------------------

    predictions_df = test_df.copy()

    predictions_df["prediction"] = y_pred
    predictions_df["probability_real"] = probabilities[:, 0]
    predictions_df["probability_fake"] = probabilities[:, 1]

    predictions_path = results_dir / "test_predictions.csv"

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Metric artifacts
    # ---------------------------------------------------------

    metrics_path = results_dir / "evaluation_metrics.json"

    evaluator.save_metrics(
        metrics=metrics,
        output_path=metrics_path,
    )

    report_path = results_dir / "classification_report.csv"

    evaluator.save_classification_report(
        report=classification_report_data,
        output_path=report_path,
    )

    # ---------------------------------------------------------
    # Visual artifacts
    # ---------------------------------------------------------

    evaluation_config = config["evaluation"]

    if evaluation_config["save_confusion_matrix"]:
        confusion_path = results_dir / "confusion_matrix.png"

        evaluator.save_confusion_matrix(
            matrix=confusion,
            output_path=confusion_path,
        )

    if evaluation_config["save_roc_curve"]:
        roc_path = results_dir / "roc_curve.png"

        evaluator.save_roc_curve(
            false_positive_rate=false_positive_rate,
            true_positive_rate=true_positive_rate,
            roc_auc=metrics["roc_auc"],
            output_path=roc_path,
        )

    # ---------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------

    logger.info("Test-set evaluation complete.")

    print("\nTest Set Metrics")
    print("=" * 40)

    for metric_name in config["evaluation"]["metrics"]:
        value = metrics[metric_name]

        print(f"{metric_name.upper():<12}: {value:.4f}")

    print("\nConfusion Matrix")
    print(confusion)

    print("\nArtifacts saved to:")
    print(results_dir)


if __name__ == "__main__":
    main()
