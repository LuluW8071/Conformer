from torch import nn


class FeedForwardBlock(nn.Module):
    """
    Conformer feed-forward block.

    Parameters:
      d_model (int): Dimension of the model
      expansion (int): Expansion factor for first linear layer
      dropout (float): Dropout probability

    Inputs:
      x (Tensor): (batch_size, time, d_model)
      mask: Unused

    Outputs:
      Tensor (batch_size, time, d_model): Output tensor from the feed-forward module

    """

    def __init__(self, d_model=144, expansion=4, dropout=0.1):
        super(FeedForwardBlock, self).__init__()
        self.module = nn.Sequential(
            nn.LayerNorm(d_model, eps=6.1e-5),
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.module(x)
