from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm


class Trainer:
    """Training and validation manager for the multimodal model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        epochs: int,
        checkpoint_dir: str | Path,
        mixed_precision: bool = True,
        gradient_clip_norm: float | None = None,
        early_stopping_patience: int = 2,
        logger: logging.Logger | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.epochs = epochs

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.mixed_precision = mixed_precision and device.type == "cuda"

        self.gradient_clip_norm = gradient_clip_norm

        self.early_stopping_patience = early_stopping_patience

        self.logger = logger if logger is not None else logging.getLogger(__name__)

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.mixed_precision,
        )

        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0

        self.history: list[dict[str, Any]] = []

    def _autocast_context(self):
        """Return the appropriate autocast context."""
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.mixed_precision,
        )

    def _forward_batch(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:

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

        with self._autocast_context():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
            )

            loss = self.criterion(
                logits,
                labels,
            )

        return loss, logits

    def train_one_epoch(
        self,
        epoch: int,
    ) -> tuple[float, float]:

        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.epochs}",
            leave=False,
        )

        for batch in progress:
            self.optimizer.zero_grad(set_to_none=True)

            loss, logits = self._forward_batch(batch)

            self.scaler.scale(loss).backward()

            if self.gradient_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.gradient_clip_norm,
                )

            self.scaler.step(self.optimizer)

            self.scaler.update()

            batch_size = batch["labels"].size(0)

            total_loss += loss.item() * batch_size

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            correct += (predictions == batch["labels"].to(self.device)).sum().item()

            total += batch_size

            progress.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = total_loss / total
        epoch_accuracy = correct / total

        return epoch_loss, epoch_accuracy

    @torch.inference_mode()
    def validate(
        self,
    ) -> tuple[float, float]:

        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.val_loader:
            loss, logits = self._forward_batch(batch)

            batch_size = batch["labels"].size(0)

            total_loss += loss.item() * batch_size

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            labels = batch["labels"].to(self.device)

            correct += (predictions == labels).sum().item()

            total += batch_size

        val_loss = total_loss / total
        val_accuracy = correct / total

        return val_loss, val_accuracy

    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        filename: str,
    ) -> None:

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
        }

        torch.save(
            checkpoint,
            self.checkpoint_dir / filename,
        )

    def fit(self) -> pd.DataFrame:

        self.logger.info(
            "Starting training for %d epochs.",
            self.epochs,
        )

        self.logger.info(
            "Device: %s",
            self.device,
        )

        self.logger.info(
            "Mixed precision: %s",
            self.mixed_precision,
        )

        for epoch in range(1, self.epochs + 1):
            train_loss, train_accuracy = self.train_one_epoch(epoch)

            val_loss, val_accuracy = self.validate()

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }

            self.history.append(record)

            self.logger.info(
                (
                    "Epoch %d | "
                    "train_loss=%.4f | "
                    "train_acc=%.4f | "
                    "val_loss=%.4f | "
                    "val_acc=%.4f"
                ),
                epoch,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
            )

            # Always save the latest state.
            self._save_checkpoint(
                epoch=epoch,
                val_loss=val_loss,
                filename="last_checkpoint.pt",
            )

            # Best-model selection.
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0

                self._save_checkpoint(
                    epoch=epoch,
                    val_loss=val_loss,
                    filename="best_model.pt",
                )

                self.logger.info(
                    "New best model saved at epoch %d.",
                    epoch,
                )

            else:
                self.patience_counter += 1

                self.logger.info(
                    "No validation improvement. Patience: %d/%d",
                    self.patience_counter,
                    self.early_stopping_patience,
                )

                if self.patience_counter >= self.early_stopping_patience:
                    self.logger.info("Early stopping triggered.")
                    break

        history_df = pd.DataFrame(self.history)

        self.logger.info(
            "Training complete. Best epoch: %d",
            self.best_epoch,
        )

        return history_df
