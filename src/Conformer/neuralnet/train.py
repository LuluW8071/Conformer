import comet_ml
import os 
import argparse
import pytorch_lightning as pl
import torch

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CometLogger
from torchmetrics.text import WordErrorRate, CharErrorRate

# Load API
from dotenv import load_dotenv
load_dotenv()

from dataloader import SpeechDataModule
from model import ConformerASR
from utils import GreedyDecoder

class ASRTrainer(pl.LightningModule):
    def __init__(self, model, args):
        super(ASRTrainer, self).__init__()
        self.model = model
        self.args = args

        # Metrics
        self.losses = []
        self.val_wer = []
        
        # NOTE: CER is computed only for test_dataloader as it is time and resource consuming for every validation step
        self.char_error_rate = CharErrorRate()
        self.word_error_rate = WordErrorRate()

        self.loss_fn = nn.CTCLoss(blank=28, zero_infinity=True)

        # Precompute sync_dist for distributed GPUs train
        self.sync_dist = True if args.gpus > 1 else False

    def forward(self, x, mask):
        return self.model(x, mask)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.8,           # Reduce LR by multiplying it by 0.8
                patience=1,           # No. of epochs to wait before reducing LR
                threshold=4e-2,       # Minimum change in val_loss to qualify as improvement
                threshold_mode='rel', # Relative threshold (e.g., 0.1% change)
                min_lr=3e-6           # Minm. LR to stop reducing
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }

        return [optimizer], [scheduler]

    def _common_step(self, batch, batch_idx):
        spectrograms, labels, input_lengths, label_lengths, mask = batch

        # Directly calls forward method of conformer and pass spectrograms and mask
        output = self(spectrograms, mask)
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

        # Log some predictions during validation phase in CometML
        # NOTE: If validation set is too less, set batch_idx % 200 or any other condition  
        # Log final predictions
        if batch_idx % 1000 == 0:
            self._text_logger(decoded_preds, decoded_targets, "Validation")

        # Calculate metrics
        wer_batch = self.word_error_rate(decoded_preds, decoded_targets)
        self.val_wer.append(wer_batch)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        # Calculate averages of metrics over the entire epoch
        avg_loss = torch.stack(self.losses).mean()
        avg_wer = torch.stack(self.val_wer).mean()

        # Log all metrics using log_dict
        metrics = {
            'val_loss': avg_loss,
            'val_wer': avg_wer
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
        self.val_wer.clear()


    def test_step(self, batch, batch_idx):
        loss, y_pred, labels, label_lengths = self._common_step(batch, batch_idx)
        self.losses.append(loss)

        # Greedy decoding
        decoded_preds, decoded_targets = GreedyDecoder(y_pred.transpose(0, 1), labels, label_lengths)
        
        # Log final predictions
        if batch_idx % 5 == 0:
            self._text_logger(decoded_preds, decoded_targets, "Test")

        # Calculate metrics
        cer_batch = self.char_error_rate(decoded_preds, decoded_targets)
        wer_batch = self.word_error_rate(decoded_preds, decoded_targets)
        
        # Log all metrics using log_dict
        metrics = {
            'test_loss': loss,
            'test_cer': cer_batch,
            'test_wer': wer_batch
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

        return metrics

    def _text_logger(self, decoded_preds, decoded_targets, phase):
        formatted_log = []

        for i in range(len(decoded_targets)):
            formatted_log.append(f"{decoded_targets[i]}\n{decoded_preds[i]}")
        log_text = "\n\n".join(formatted_log)
        self.logger.experiment.log_text(text=log_text, metadata={"Phase": phase})


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    directory = "data"
    if not os.path.exists(directory):
        os.makedirs(directory)

    data_module = SpeechDataModule(
        batch_size=args.batch_size,
        train_url=[
            "train-clean-100", 
            "train-clean-360", 
            "train-other-500"
        ],
        valid_url=[
            "test-clean", 
            "test-other"
        ],
        train_json=args.train_json,
        test_json=args.valid_json,
        num_workers=args.num_workers)

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
    # NOTE: Commented out since model compilation got slower instead
    # model = torch.compile(model)      

    speech_trainer = ASRTrainer(model=model, args=args)

    # NOTE: Comet Logger
    comet_logger = CometLogger(
        api_key=os.getenv("API_KEY"), project_name=os.getenv("PROJECT_NAME")
    )

    # NOTE: Define Trainer callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./saved_checkpoint/",
        filename="Conformer-{epoch:02d}-{val_wer:.3f}",
        save_top_k=3,   # 3 Checkpoints
        mode="min",
    )

    # Trainer Instance
    trainer_args = {
        'accelerator': device,                                          # Device to use for training
        'devices': args.gpus,                                           # Number of GPUs to use for training
        'min_epochs': 1,                                                # Minm. no. of epochs to run
        'max_epochs': args.epochs,                                      # Maxm. no. of epochs to run                               
        'precision': args.precision,                                    # Precision to use for training
        'check_val_every_n_epoch': 1,                                   # No. of epochs to run validation
        'gradient_clip_val': args.grad_clip,                            # Gradient norm clipping value
        'callbacks': [LearningRateMonitor(logging_interval='epoch'),    # Callbacks to use for training
                      EarlyStopping(monitor="val_loss", patience=3),
                      checkpoint_callback],
        'logger': comet_logger,                                         # Logger to use for training
    }

    if args.accumulate_grad > 1:
        trainer_args['accumulate_grad_batches'] = args.accumulate_grad

    trainer = pl.Trainer(**trainer_args)

    # Train and Validate
    trainer.fit(speech_trainer, data_module, ckpt_path=args.checkpoint_path)
    trainer.validate(speech_trainer, data_module)
    trainer.fit(speech_trainer, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASR Model")

    # Train Device Hyperparameters
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-w', '--num_workers', default=8, type=int, help='n data loading workers')
    parser.add_argument('-db', '--dist_backend', default='ddp', type=str, help='which distributed backend to use for aggregating multi-gpu train')

    # Train and Valid File
    parser.add_argument('--train_json', default=None, required=True, type=str, help='json file to load training data')                   
    parser.add_argument('--valid_json', default=None, required=True, type=str, help='json file to load testing data')

    # General Train Hyperparameters
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, help='size of batch')
    parser.add_argument('-lr','--learning_rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--precision', default='16-mixed', type=str, help='precision')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='path of checkpoint file to resume training')
    parser.add_argument('-gc', '--grad_clip', default=1.0, type=float, help='gradient norm clipping value')
    parser.add_argument('-ag', '--accumulate_grad', default=1, type=int, help='number of batches to accumulate gradients over')

    args = parser.parse_args()
    main(args)
