from torch import nn


class ConvBlock(nn.Module):
    """
    Conformer convolutional block.

    Parameters:
      d_model (int): Dimension of the model
      kernel_size (int): Size of kernel to use for depthwise convolution
      dropout (float): Dropout probability

    Inputs:
      x (Tensor): (batch_size, time, d_model)
      mask: Unused

    Outputs:
      Tensor (batch_size, time, d_model): Output tensor from the convolution module

    """

    def __init__(self, d_model=144, kernel_size=31, dropout=0.1):
        super(ConvBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=6.1e-5)
        kernel_size = 31
        self.module = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=kernel_size,
                padding="same",
                groups=d_model,
            ),
            nn.BatchNorm1d(d_model, eps=6.1e-5),
            nn.SiLU(),
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.module(x)
        return x.transpose(1, 2)
