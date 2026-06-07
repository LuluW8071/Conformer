from torch import nn


class Conv2dSubsampling(nn.Module):
  """
    2d Convolutional subsampling.
    Subsamples time and freq domains of input spectrograms by a factor of 4, d_model times.

    Parameters:
      d_model (int): Dimension of the model

    Inputs:
      x (Tensor): Input spectrogram (batch_size, time, d_input)

    Outputs:
      Tensor (batch_size, time, d_model * (d_input // 4)): Output tensor from the conlutional subsampling module

  """

  def __init__(self, d_model=144):
    super(Conv2dSubsampling, self).__init__()
    self.module = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=d_model,
                  kernel_size=3, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=d_model, out_channels=d_model,
                  kernel_size=3, stride=2),
        nn.ReLU(),
    )

  def forward(self, x):
    output = self.module(x.unsqueeze(1))
    batch_size, d_model, subsampled_time, subsampled_freq = output.size()
    output = output.permute(0, 2, 1, 3)
    output = output.contiguous().view(
        batch_size, subsampled_time, d_model * subsampled_freq)
    return output
