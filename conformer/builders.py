"""
Factory helpers that assemble the data module, model, callbacks, logger and Trainer.
"""

from __future__ import annotations

import os
from typing import List, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CometLogger

from conformer.config import ModelConfig
from conformer.model import ConformerASR
from conformer.data.dataloader import SpeechDataModule


def build_data_module(args) -> SpeechDataModule:
    data_module = SpeechDataModule(
        batch_size=args.batch_size,
        train_url=["train-clean-100", "train-clean-360", "train-other-500"],
        valid_url=["test-clean", "test-other"],
        train_json=args.train_json,
        test_json=args.valid_json,
        num_workers=args.num_workers,
    )
    data_module.setup()
    return data_module


def build_model(config: Optional[ModelConfig] = None) -> ConformerASR:
    config = config or ModelConfig()
    return ConformerASR(config.encoder.to_dict(), config.decoder.to_dict())


def build_callbacks(
    early_stopping_patience: int = 3,
    save_top_k: int = 3,
    checkpoint_dir: str = "./saved_checkpoint/",
) -> List[Callback]:
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_dir,
        filename="Conformer-{epoch:02d}-{val_wer:.3f}",
        save_top_k=save_top_k,
        mode="min",
    )
    return [
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(monitor="val_loss", patience=early_stopping_patience),
        checkpoint_callback,
    ]


def build_logger() -> CometLogger:
    return CometLogger(
        api_key=os.getenv("API_KEY"),
        project=os.getenv("PROJECT_NAME"),
    )


def build_trainer(args, accelerator: str) -> pl.Trainer:
    trainer_args = {
        "accelerator": accelerator,
        "devices": args.gpus,
        "min_epochs": 1,
        "max_epochs": args.epochs,
        "precision": args.precision,
        "check_val_every_n_epoch": 1,
        "accumulate_grad_batches": args.accumulate_grad,
        "gradient_clip_val": args.grad_clip,
        "callbacks": build_callbacks(),
        "logger": build_logger(),
    }
    if args.gpus > 1:
        trainer_args["strategy"] = args.dist_backend
    return pl.Trainer(**trainer_args)