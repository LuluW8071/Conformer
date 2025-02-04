import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchaudio.datasets import LIBRISPEECH

from torch.utils.data import DataLoader, ConcatDataset
from dataset import LibriSpeechDataset, MozillaDataset

# Lightning Data Module
class SpeechDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_url, valid_url, train_json, test_json, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.train_url = train_url
        self.valid_url = valid_url
        self.train_json = train_json
        self.test_json = test_json
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Load LibriSpeech datasets
        train_librispeech = [LIBRISPEECH("./data", url=url, download=True) for url in self.train_url]
        valid_librispeech = [LIBRISPEECH("./data", url=url, download=True) for url in self.valid_url]

        # Load Mozilla Common Voice datasets
        train_mozilla = MozillaDataset(self.train_json, valid=False)
        valid_mozilla = MozillaDataset(self.test_json, valid=True)

        # Combine datasets
        combined_train_dataset = ConcatDataset([LibriSpeechDataset(ConcatDataset(train_librispeech)), train_mozilla])
        combined_valid_dataset = ConcatDataset([LibriSpeechDataset(ConcatDataset(valid_librispeech)), valid_mozilla])

        self.train_dataset = combined_train_dataset
        self.valid_dataset = combined_valid_dataset
        self.test_dataset = valid_librispeech

    def data_processing(self, data):
        spectrograms, labels, input_lengths, label_lengths = [],[],[],[]
        for (spectrogram, label, label_length) in data:
            if spectrogram is None:
                continue
            spectrograms.append(spectrogram.squeeze(0).transpose(0, 1))
            labels.append(torch.Tensor(label))
            input_lengths.append(((spectrogram.shape[-1] - 1) // 2 - 1) // 2)
            label_lengths.append(label_length)

        # Pad the spectrograms to have the same width (time dimension)
        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

        # Convert input_lengths and label_lengths to tensors
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)
        label_lengths = torch.tensor(label_lengths, dtype=torch.long)

        mask = torch.ones(
            spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[1]
        )
        for i, l in enumerate(input_lengths):
            mask[i, :, :l] = 0

        return spectrograms, labels, input_lengths, label_lengths, mask.bool()
        

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda x: self.data_processing(x),
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: self.data_processing(x),
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: self.data_processing(x),
            num_workers=self.num_workers,
            pin_memory=True,
        )
