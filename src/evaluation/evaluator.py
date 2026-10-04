from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm


class Evaluator:
    """Evaluate a trained multimodal classifier on an untouched test set."""

    def __init__(
        self,
        model: torch.nn.Module,
        test_loader,
        device: torch.device,
        logger: logging.Logger | None = None,
    ) -> None:
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    @torch.inference_mode()
    def predict(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate predictions and class probabilities for the test set."""

        self.model.eval()

        all_labels: list[np.ndarray] = []
        all_predictions: list[np.ndarray] = []
        all_probabilities: list[np.ndarray] = []

        progress = tqdm(
            self.test_loader,
            desc="Evaluating test set",
            leave=False,
        )

        for batch in progress:
            input_ids = batch["input_ids"].to(
                self.device,
                non_blocking=True,
            )

            attention_mask = batch["attention_mask"].to(
                self.device,
                non_blocking=True,
            )

            images = batch["images"].to(
                self.device,
                non_blocking=True,
            )

            labels = batch["labels"].to(
                self.device,
                non_blocking=True,
            )

            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            all_labels.append(
                labels.cpu().numpy(),
            )

            all_predictions.append(
                predictions.cpu().numpy(),
            )

            all_probabilities.append(
                probabilities.cpu().numpy(),
            )

        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_predictions)
        probabilities = np.concatenate(all_probabilities)

        self.logger.info(
            "Collected predictions for %d test samples.",
            len(y_true),
        )

        return y_true, y_pred, probabilities

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        probabilities: np.ndarray,
    ) -> tuple[
        dict[str, float],
        dict[str, Any],
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Compute classification metrics and ROC/CM data."""

        # Class 1 = fake.
        fake_probabilities = probabilities[:, 1]

        metrics = {
            "accuracy": float(
                accuracy_score(y_true, y_pred),
            ),
            "precision": float(
                precision_score(
                    y_true,
                    y_pred,
                    pos_label=1,
                    zero_division=0,
                ),
            ),
            "recall": float(
                recall_score(
                    y_true,
                    y_pred,
                    pos_label=1,
                    zero_division=0,
                ),
            ),
            "f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    pos_label=1,
                    zero_division=0,
                ),
            ),
            "roc_auc": float(
                roc_auc_score(
                    y_true,
                    fake_probabilities,
                ),
            ),
        }

        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["real", "fake"],
            output_dict=True,
            zero_division=0,
        )

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )

        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_true,
            fake_probabilities,
            pos_label=1,
        )

        return (
            metrics,
            report,
            matrix,
            false_positive_rate,
            true_positive_rate,
        )

    @staticmethod
    def save_metrics(
        metrics: dict[str, float],
        output_path: str | Path,
    ) -> None:
        """Save scalar evaluation metrics as JSON."""

        output_path = Path(output_path)

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metrics,
                file,
                indent=2,
            )

    @staticmethod
    def save_classification_report(
        report: dict[str, Any],
        output_path: str | Path,
    ) -> None:
        """Save class-wise classification report as CSV."""

        report_df = pd.DataFrame(report).transpose()

        report_df.to_csv(
            output_path,
            index=True,
        )

    @staticmethod
    def save_confusion_matrix(
        matrix: np.ndarray,
        output_path: str | Path,
    ) -> None:
        """Save confusion matrix visualization."""

        figure, axis = plt.subplots(
            figsize=(6, 5),
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["Real", "Fake"],
        )

        display.plot(
            ax=axis,
            values_format="d",
        )

        axis.set_title("Confusion Matrix")
        axis.set_xlabel("Predicted Label")
        axis.set_ylabel("True Label")

        figure.tight_layout()

        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)

    @staticmethod
    def save_roc_curve(
        false_positive_rate: np.ndarray,
        true_positive_rate: np.ndarray,
        roc_auc: float,
        output_path: str | Path,
    ) -> None:
        """Save ROC curve visualization."""

        figure, axis = plt.subplots(
            figsize=(7, 5),
        )

        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"ROC-AUC = {roc_auc:.4f}",
        )

        axis.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Random Classifier",
        )

        axis.set_title("ROC Curve")
        axis.set_xlabel("False Positive Rate")
        axis.set_ylabel("True Positive Rate")
        axis.legend(loc="lower right")

        figure.tight_layout()

        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)
