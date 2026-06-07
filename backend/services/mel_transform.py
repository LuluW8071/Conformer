import torch.nn as nn
from torchaudio import transforms as T

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

