import comet_ml
import os 
import argparse
import pytorch_lightning as pl
import torch

import torch.nn.functional as F
import torch.optim as optim
import torch import nn

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CometLogger
from torchmetrics.text import WordErrorRate, CharErrorRate

# Load API
from dotenv import load_dotenv
load_dotenv()

from dataset import SpeechDataModule
from Conformer import ConformerEncoder, LSTMDecoder
from utils import GreedyDecoder


class ConformerASR(nn.Module):
    def __init__(self, encoder_params, decoder_params):
        super(ConformerASR, self).__init__()
        self.encoder = ConformerEncoder(**encoder_params)
        self.decoder = LSTMDecoder(**decoder_params)

    def forward(self, x):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return decoder_output


class ASRTrainer(pl.LightningModule):
    def __init__(self, model, args):
        super(ASRTrainer, self).__init__()
        self.model = model
        self.args = args

        self.losses = []
        self.val_cer = []
        self.val_wer = []

        # Metrics
        # NOTE: Comment CER since validation phase takes a lot of time to compute error for each character
        self.char_error_rate = CharErrorRate()
        self.word_error_rate = WordErrorRate()

        self.loss_fn = nn.CTCLoss(blank=28, zero_infinity=True)

        # Precompute sync_dist for distributed GPUs train
        self.sync_dist = True if args.gpus > 1 else False

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

        scheduler = {
            "scheduler": optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=5,             # Number of epochs for the first restart
                T_mult=2,           # Factor to increase T_0 after each restart
                eta_min=5e-6,       # Minimum learning rate
            ),
            "monitor": "val_loss",
        }

        return [optimizer], [scheduler]

    def _common_step(self, batch, batch_idx):
        spectrograms, labels, input_lengths, label_lengths, _ = batch

        # Directly calls forward method of conformer and pass spectrograms
        output = self(spectrograms)
        output = F.log_softmax(output, dim=-1).transpose(0, 1)

        # Compute CTC loss
        loss = self.loss_fn(output, labels, input_lengths, label_lengths)
        return loss, output, labels, label_lengths

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self._common_step(batch, batch_idx)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=self.sync_dist,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_pred, labels, label_lengths = self._common_step(batch, batch_idx)
        self.losses.append(loss)

        # Greedy decoding
        decoded_preds, decoded_targets = GreedyDecoder(y_pred.transpose(0, 1), labels, label_lengths)

        # Calculate metrics
        cer_batch = self.char_error_rate(decoded_preds, decoded_targets)
        wer_batch = self.word_error_rate(decoded_preds, decoded_targets)

        # Append batch metrics to lists
        self.val_cer.append(cer_batch)
        self.val_wer.append(wer_batch)

        # Log some predictions during validation phase in CometML
        # NOTE: If validation set is too less, set batch_idx % 20 or any other condition  
        if batch_idx % 200 == 0:
            log_targets = decoded_targets[0]
            log_preds = {"Preds": decoded_preds[0]}
            self.logger.experiment.log_text(text=log_targets, metadata=log_preds)

        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        # Calculate averages of metrics over the entire epoch
        avg_loss = torch.stack(self.losses).mean()
        avg_cer = torch.stack(self.val_cer).mean()
        avg_wer = torch.stack(self.val_wer).mean()

        # Prepare metrics dictionary and log all metrics at once
        metrics = {
            "val_cer": avg_cer,
            "val_wer": avg_wer,
            "val_loss": avg_loss,
        }

        # Log all metrics at once
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.args.batch_size,
            sync_dist=self.sync_dist,
        )

        # Clear the lists for the next epoch
        self.losses.clear()
        self.val_cer.clear()
        self.val_wer.clear()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare dataset
    data_module = SpeechDataModule(
        batch_size=args.batch_size,
        train_json=args.train_json,
        test_json=args.valid_json,
        num_workers=args.num_workers,
    )
    data_module.setup()

    # Define model hyperparameters
    # https://arxiv.org/pdf/2005.08100 : Table 1 for conformer parameters
    encoder_params = {
        "d_input": 80,          # Input features: n-mels
        "d_model": 144,         # Encoder Dims
        "num_layers": 16,       # Encoder Layers
        "conv_kernel_size": 31,
        "feed_forward_residual_factor": 0.5,
        "feed_forward_expansion_factor": 4,
        "num_heads": 4,         # Relative MultiHead Attetion Heads
        "dropout": 0.1,
    }

    decoder_params = {
        "d_encoder": 144,       # Match with Encoder layer
        "d_decoder": 320,       # Decoder Dim
        "num_layers": 1,        # Deocder Layer
        "num_classes": 29,      # Output Classes
    }

    # Optimize Model Instance for faster training
    model = ConformerASR(encoder_params, decoder_params)
    model = torch.compile(model)

    speech_trainer = ASRTrainer(model=model, args=args)

    # NOTE: Comet Logger
    comet_logger = CometLogger(
        api_key=os.getenv("API_KEY"), project_name=os.getenv("PROJECT_NAME")
    )

    # NOTE: Define Trainer callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./saved_checkpoint/",
        filename="Conformer-{epoch:02d}-{val_loss:.2f}-{val_wer:.2f}",
        save_top_k=3,   # 3 Checkpoints
        mode="min",
    )

    # Trainer Instance
    trainer_args = {
        "accelerator": device,
        "devices": args.gpus,
        "strategy": args.dist_backend if args.gpus > 1 else "auto",   # Distributed Backend for multi GPU training
        "min_epochs": 1,
        "max_epochs": args.epochs,
        "precision": args.precision,
        "check_val_every_n_epoch": 1,
        "gradient_clip_val": 1.0,
        "callbacks": [
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(monitor="val_loss", patience=5),  # Early stopping
            checkpoint_callback,
        ],
        "logger": comet_logger,

    }

    trainer = pl.Trainer(**trainer_args)

    # Train and Validate
    trainer.fit(speech_trainer, data_module, ckpt_path=args.checkpoint_path)
    trainer.validate(speech_trainer, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASR Model")

    # Train Device Hyperparameters
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-w', '--num_workers', default=8, type=int, help='n data loading workers')
    parser.add_argument('-db', '--dist_backend', default='ddp', type=str,
                        help='which distributed backend to use for aggregating multi-gpu train')

    # Train and Valid File
    parser.add_argument('--train_json', default=None, required=True, type=str, help='json file to load training data')                   
    parser.add_argument('--valid_json', default=None, required=True, type=str, help='json file to load testing data')

    # General Train Hyperparameters
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=32, type=int, help='size of batch')
    parser.add_argument('-lr', '--learning_rate', default=4e-5, type=float, help='learning rate')
    parser.add_argument('--precision', default='16-mixed', type=str, help='precision')
    
    # Checkpoint path
    parser.add_argument('--checkpoint_path', default=None, type=str, help='path to a checkpoint file to resume training')

    args = parser.parse_args()
    main(args)
