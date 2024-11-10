import pytorch_lightning as pl
import json
import torchaudio
import torch
import torch.nn as nn
import torchaudio.transforms as transforms

from torch.utils.data import DataLoader, Dataset
from utils import TextTransform  # Comment this for engine inference


class MelSpec(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80, hop_length=160):
        super(MelSpec, self).__init__()
        self.transform = transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, hop_length=hop_length
        )

    def forward(self, x):
        return self.transform(x)


# For Engine Inference Only
def get_featurizer(sample_rate=16000, n_mels=80, hop_length=160):
    return MelSpec(sample_rate=sample_rate, n_mels=n_mels, hop_length=hop_length)


# Custom Dataset Class
class CustomAudioDataset(Dataset):
    def __init__(self, json_path, log_ex=True, valid=False):
        print(f"Loading json data from {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        # print(self.data)
        self.text_process = TextTransform()
        self.log_ex = log_ex

        if valid:
            self.audio_transforms = torch.nn.Sequential(MelSpec())
        else:
            # Time & Frequency Masking
            time_masks = [
                torchaudio.transforms.TimeMasking(time_mask_param=15, p=0.05)
                for _ in range(10)
            ]
            self.audio_transforms = nn.Sequential(
                MelSpec(),
                transforms.FrequencyMasking(freq_mask_param=27),
                *time_masks,
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        file_path = item["key"]

        try:
            waveform, _ = torchaudio.load(file_path)             # Point to location of audio data
            utterance = item["text"].lower()                     # Point to sentence of audio data
            label = self.text_process.text_to_int(utterance)     # Convert characters to integer map from utils.py
            spectrogram = self.audio_transforms(waveform)        # (channel, feature, time)
            label_len = len(label)

            if spectrogram.shape[0] > 1:
                raise Exception("dual channel, skipping audio file %s" % file_path)
            if spectrogram.shape[2] > 4096 * 2:
                raise Exception("spectrogram too big. size %s" % spectrogram.shape[2])
            if label_len == 0:
                raise Exception("label len is zero... skipping %s" % file_path)

            return spectrogram, label, label_len

        except Exception as e:
            if self.log_ex:
                print(f"{str(e)}\r", end="")
            return self.__getitem__(idx - 1 if idx != 0 else idx + 1)

    def describe(self):
        return self.data.describe()


# Lightning Data Module
class SpeechDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_json, test_json, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.train_json = train_json
        self.test_json = test_json
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = CustomAudioDataset(self.train_json, valid=False)
        self.test_dataset = CustomAudioDataset(self.test_json, valid=True)

    def data_processing(self, data):
        spectrograms, labels, input_lengths, label_lengths = [],[],[],[]
        for spectrogram, label, label_length in data:
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
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: self.data_processing(x),
            num_workers=self.num_workers,
            pin_memory=True,
        )
