import torch
from torch import nn


class PositionalEncoder(nn.Module):
    """
    Generate positional encodings used in the relative multi-head attention module.
    Same encodings as the original transformer model [Attention Is All You Need]:
    https://arxiv.org/abs/1706.03762

    Parameters:
      max_len (int): Maximum sequence length (time dimension)

    Inputs:
      len (int): Length of encodings to retrieve

    Outputs
      Tensor (len, d_model): Positional encodings
    """

    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        encodings = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float)
        inv_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        encodings[:, 0::2] = torch.sin(pos[:, None] * inv_freq)
        encodings[:, 1::2] = torch.cos(pos[:, None] * inv_freq)
        self.register_buffer("encodings", encodings)

    def forward(self, len):
        return self.encodings[:len, :]
