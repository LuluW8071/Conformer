"""Entry point for training the Conformer ASR model.

Run:
    python train.py --train_json path/to/train.json --valid_json path/to/valid.json
"""
import os
import argparse

import comet_ml  # noqa: F401  (imported first so Comet can auto-instrument)
import torch
import pytorch_lightning as pl
from dotenv import load_dotenv

from conformer.config import ModelConfig
from conformer.asr_trainer import ASRTrainer
from conformer.builders import build_data_module, build_model, build_trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ASR Model")

    # Train device hyperparameters
    parser.add_argument("-g", "--gpus", default=1, type=int,
                        help="number of gpus per node")
    parser.add_argument("-w", "--num_workers", default=8, type=int,
                        help="n data loading workers")
    parser.add_argument("-db", "--dist_backend", default="deepspeed_stage_2", type=str,
                        help="distributed backend for multi-gpu training")

    # Train and valid files
    parser.add_argument("--train_json", required=True, type=str,
                        help="json file to load training data")
    parser.add_argument("--valid_json", required=True, type=str,
                        help="json file to load validation data")

    # General train hyperparameters
    parser.add_argument("--epochs", default=50, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="size of batch")
    parser.add_argument("-lr", "--learning_rate", default=1e-4, type=float,
                        help="learning rate")
    parser.add_argument("--precision", default="16-mixed", type=str,
                        help="precision")
    parser.add_argument("--checkpoint_path", default=None, type=str,
                        help="path of checkpoint file to resume training")
    parser.add_argument("-gc", "--grad_clip", default=1.0, type=float,
                        help="gradient norm clipping value")
    parser.add_argument("-ag", "--accumulate_grad", default=1, type=int,
                        help="number of batches to accumulate gradients over")

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    load_dotenv()
    os.makedirs("data", exist_ok=True)

    accelerator = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    data_module = build_data_module(args)

    # Model + LightningModule
    model_config = ModelConfig()
    model = build_model(model_config)
    lightning_module = ASRTrainer(
        model=model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_gpus=args.gpus,
        blank_id=model_config.blank_id,
    )

    # Trainer
    trainer = build_trainer(args, accelerator)

    # Train (resuming from a checkpoint if one was provided), then validate.
    trainer.fit(lightning_module, data_module, ckpt_path=args.checkpoint_path)
    trainer.validate(lightning_module, data_module)


if __name__ == "__main__":
    main(parse_args())