"""
PyTorch Lightning module wrapping the Conformer ASR model.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.text import WordErrorRate, CharErrorRate

from conformer.utils import GreedyDecoder


class ASRTrainer(pl.LightningModule):
    """Trains / validates / tests the Conformer encoder-decoder with CTC loss."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        batch_size: int = 64,
        num_gpus: int = 1,
        blank_id: int = 28,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.blank_id = blank_id

        # Persist scalar hyperparameters into the checkpoint (everything except
        # the model object, which is reconstructed separately).
        self.save_hyperparameters(ignore=["model"])

        # Metrics. CER is only computed at test time as it is expensive to run
        # on every validation step.
        self.char_error_rate = CharErrorRate()
        self.word_error_rate = WordErrorRate()
        self.loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True)

        # Buffers for epoch-level metric aggregation.
        self._losses: List[torch.Tensor] = []
        self._wers: List[torch.Tensor] = []

        # Only sync metrics across processes when training on multiple GPUs.
        self.sync_dist = num_gpus > 1

    # ------------------------------------------------------------------ #
    # Core
    # ------------------------------------------------------------------ #
    def forward(self, x, mask):
        return self.model(x, mask)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
        scheduler = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.8,            # reduce LR by multiplying by 0.8
                patience=1,            # epochs to wait before reducing LR
                threshold=4e-2,        # min relative change counted as improvement
                threshold_mode="rel",
                min_lr=3e-6,           # floor LR
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def _common_step(self, batch, batch_idx):
        spectrograms, labels, input_lengths, label_lengths, mask = batch

        output = self(spectrograms, mask)                       # (B, T, C)
        output = F.log_softmax(output, dim=-1).transpose(0, 1)  # (T, B, C) for CTC
        loss = self.loss_fn(output, labels, input_lengths, label_lengths)
        return loss, output, labels, label_lengths

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def training_step(self, batch, batch_idx):
        loss, *_ = self._common_step(batch, batch_idx)
        self.log(
            "train_loss", loss,
            on_step=True, on_epoch=False, prog_bar=True, logger=True,
            sync_dist=self.sync_dist,
        )
        return loss

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validation_step(self, batch, batch_idx):
        loss, y_pred, labels, label_lengths = self._common_step(batch, batch_idx)
        self._losses.append(loss)

        decoded_preds, decoded_targets = GreedyDecoder(
            y_pred.transpose(0, 1), labels, label_lengths
        )

        # Log a sample of predictions. If the validation set is small, lower
        # this interval (e.g. `batch_idx % 200`) to see more examples.
        if batch_idx % 1000 == 0:
            self._log_predictions(decoded_preds, decoded_targets, "Validation")

        self._wers.append(self.word_error_rate(decoded_preds, decoded_targets))
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        metrics = {
            "val_loss": torch.stack(self._losses).mean(),
            "val_wer": torch.stack(self._wers).mean(),
        }
        self.log_dict(
            metrics,
            on_step=False, on_epoch=True, prog_bar=True, logger=True,
            batch_size=self.batch_size, sync_dist=self.sync_dist,
        )
        self._losses.clear()
        self._wers.clear()

    # ------------------------------------------------------------------ #
    # Test
    # ------------------------------------------------------------------ #
    def test_step(self, batch, batch_idx):
        loss, y_pred, labels, label_lengths = self._common_step(batch, batch_idx)

        decoded_preds, decoded_targets = GreedyDecoder(
            y_pred.transpose(0, 1), labels, label_lengths
        )
        if batch_idx % 5 == 0:
            self._log_predictions(decoded_preds, decoded_targets, "Test")

        metrics = {
            "test_loss": loss,
            "test_cer": self.char_error_rate(decoded_preds, decoded_targets),
            "test_wer": self.word_error_rate(decoded_preds, decoded_targets),
        }
        self.log_dict(
            metrics,
            on_step=False, on_epoch=True, prog_bar=True, logger=True,
            batch_size=self.batch_size, sync_dist=self.sync_dist,
        )
        return metrics

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _log_predictions(self, decoded_preds, decoded_targets, phase: str) -> None:
        formatted = [
            f"{target}\n{pred}"
            for target, pred in zip(decoded_targets, decoded_preds)
        ]
        self.logger.experiment.log_text(
            text="\n\n".join(formatted), metadata={"Phase": phase}
        )