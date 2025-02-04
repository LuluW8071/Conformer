import math
import torch
import torch.nn.functional as F
import torch.nn as nn

from .encoding import PositionalEncoder


class RelativeMultiHeadAttention(nn.Module):
    """
    Relative Multi-Head Self-Attention Module.
    Method proposed in Transformer-XL paper: https://arxiv.org/abs/1901.02860

    Parameters:
      d_model (int): Dimension of the model
      num_heads (int): Number of heads to split inputs into
      dropout (float): Dropout probability
      positional_encoder (nn.Module): PositionalEncoder module

    Inputs:
      x (Tensor): (batch_size, time, d_model)
      mask (Tensor): (batch_size, time, time) Optional mask to zero out attention score at certain indices

    Outputs:
      Tensor (batch_size, time, d_model): Output tensor from the attention module.

    """

    def __init__(
        self,
        d_model=144,
        num_heads=4,
        dropout=0.1,
        positional_encoder=PositionalEncoder(144),
    ):
        super(RelativeMultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0
        self.d_model = d_model
        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_pos = nn.Linear(d_model, d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model)

        self.u = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u)
        torch.nn.init.xavier_uniform_(self.v)

        self.layer_norm = nn.LayerNorm(d_model, eps=6.1e-5)
        self.positional_encoder = positional_encoder
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()

        x = self.layer_norm(x)
        pos_emb = self.positional_encoder(seq_length)
        pos_emb = pos_emb.repeat(batch_size, 1, 1)

        q = self.W_q(x).view(batch_size, seq_length, self.num_heads, self.d_head)
        k = (
            self.W_k(x)
            .view(batch_size, seq_length, self.num_heads, self.d_head)
            .permute(0, 2, 3, 1)
        )
        v = (
            self.W_v(x)
            .view(batch_size, seq_length, self.num_heads, self.d_head)
            .permute(0, 2, 3, 1)
        )
        pos_emb = (
            self.W_pos(pos_emb)
            .view(batch_size, -1, self.num_heads, self.d_head)
            .permute(0, 2, 3, 1)
        )

        AC = torch.matmul((q + self.u).transpose(1, 2), k)
        BD = torch.matmul((q + self.v).transpose(1, 2), pos_emb)
        BD = self.rel_shift(BD)
        attn = (AC + BD) / math.sqrt(self.d_model)

        if mask is not None:
            mask = mask.unsqueeze(1)
            mask_value = -1e30 if attn.dtype == torch.float32 else -1e4
            attn.masked_fill_(mask, mask_value)

        attn = F.softmax(attn, -1)

        output = torch.matmul(attn, v.transpose(2, 3)).transpose(1, 2)
        output = output.contiguous().view(batch_size, -1, self.d_model)
        output = self.W_out(output)
        return self.dropout(output)

    def rel_shift(self, emb):
        """
        Pad and shift form relative positional encodings.
        Taken from Transformer-XL implementation: https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py
        """
        batch_size, num_heads, seq_length1, seq_length2 = emb.size()
        zeros = emb.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_emb = torch.cat([zeros, emb], dim=-1)
        padded_emb = padded_emb.view(
            batch_size, num_heads, seq_length2 + 1, seq_length1
        )
        shifted_emb = padded_emb[:, :, 1:].view_as(emb)
        return shifted_emb
