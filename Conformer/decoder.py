""" Reference: https://arxiv.org/abs/2005.08100 """

from torch import nn


class LSTMDecoder(nn.Module):
    """
    LSTM Decoder

    Parameters:
      d_encoder (int): Output dimension of the encoder
      d_decoder (int): Hidden dimension of the decoder
      num_layers (int): Number of LSTM layers to use in the decoder
      num_classes (int): Number of output classes to predict

    Inputs:
      x (Tensor): (batch_size, time, d_encoder)

    Outputs:
      Tensor (batch_size, time, num_classes): Class prediction logits

    """

    def __init__(self, d_encoder=144, d_decoder=320, num_layers=1, num_classes=29):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=d_encoder,
            hidden_size=d_decoder,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(d_decoder, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        logits = self.linear(x)
        return logits
