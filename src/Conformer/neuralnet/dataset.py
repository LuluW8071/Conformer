import json
import torchaudio
import torch
import torch.nn as nn
import torchaudio.transforms as T

from torch.utils.data import Dataset
from utils import TextTransform  # Comment this for engine inference


class MelSpec(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80, hop_length=160):
        super(MelSpec, self).__init__()
        self.transform = T.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, hop_length=hop_length
        )

    def forward(self, x):
        return self.transform(x)


# For Engine Inference Only
def get_featurizer(sample_rate=16000, n_mels=80, hop_length=160):
    return MelSpec(sample_rate=sample_rate, n_mels=n_mels, hop_length=hop_length)


class BaseSpeechDataset(Dataset):
    """ Base Speech Dataset to handle transformation of spectrograms """
    def __init__(self, log_ex=True, valid=False):
        self.text_process = TextTransform()
        self.log_ex = log_ex
        
        self.audio_transforms = self._init_transforms(valid)

    def _init_transforms(self, valid):
        if valid:
            return nn.Sequential(MelSpec())
        else:
            return nn.Sequential(
                MelSpec(),
                T.FrequencyMasking(freq_mask_param=27),
                *[T.TimeMasking(time_mask_param=15, p=0.1) for _ in range(10)]
            )

    def _process_sample(self, waveform, text, file_path=None):
        try:
            utterance = text.lower()
            label = self.text_process.text_to_int(utterance)
            spectrogram = self.audio_transforms(waveform)  # (channel, feature, time)
            label_len = len(label)

            if spectrogram.shape[0] > 1:
                raise Exception(f"dual channel, skipping audio file {file_path}")
            # Accept audio samples of only ~20 sec
            # NOTE: higher duration consumes more memory and is slower to train
            if spectrogram.shape[2] > 2048:
                raise Exception(f"spectrogram too big. size {spectrogram.shape[2]}")
            if label_len == 0:
                raise Exception(f"label len is zero... skipping {file_path}")
            
            return spectrogram, label, label_len

        except Exception as e:
            if self.log_ex:
                print(f"{str(e)}\r", end="")
            return None


class LibriSpeechDataset(BaseSpeechDataset):
    """ LibriSpeech Dataset Loader"""
    def __init__(self, dataset, log_ex=True, valid=False):
        super().__init__(log_ex, valid)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            waveform, _, utterance, _, _, _ = self.dataset[idx]
            result = self._process_sample(waveform, utterance)
            return result if result else self.__getitem__(idx - 1 if idx != 0 else idx + 1)
        except Exception as e:
            if self.log_ex:
                print(f"Error loading sample: {str(e)}\r", end="")
            return self.__getitem__(idx - 1 if idx != 0 else idx + 1)


class MozillaDataset(BaseSpeechDataset):
    """ Mozilla Corpus Dataset Loader"""
    def __init__(self, json_path, log_ex=True, valid=False):
        super().__init__(log_ex, valid)
        print(f"Loading json data from {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        file_path = item["key"]

        try:
            waveform, _ = torchaudio.load(file_path)
            result = self._process_sample(waveform, item["text"], file_path)
            return result if result else self.__getitem__(idx - 1 if idx != 0 else idx + 1)
        except Exception as e:
            if self.log_ex:
                print(f"{str(e)}\r", end="")
            return self.__getitem__(idx - 1 if idx != 0 else idx + 1)