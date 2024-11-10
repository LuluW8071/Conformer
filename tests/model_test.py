import sys
import torch
from pathlib import Path
from torch import nn

sys.path.append(str(Path(__file__).parent.parent))
from Conformer import ConformerEncoder, LSTMDecoder


class ConformerASR(nn.Module):
    def __init__(self, encoder_params, decoder_params):
        super(ConformerASR, self).__init__()
        self.encoder = ConformerEncoder(**encoder_params)
        self.decoder = LSTMDecoder(**decoder_params)

    def forward(self, x):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return decoder_output

# Reference: https://arxiv.org/pdf/2005.08100
# Appendix: Table 1 for conformer small parameters
encoder_params = {
    "d_input": 80,                          # Input features: n-mels
    "d_model": 144,                         # Encoder Dims
    "num_layers": 16,                       # Encoder Layers
    "conv_kernel_size": 31,
    "feed_forward_residual_factor": 0.5,
    "feed_forward_expansion_factor": 4,
    "num_heads": 4,                         # Relative MultiHead Attetion Heads
    "dropout": 0.1,
}

decoder_params = {
    "d_encoder": 144,                       # Match with Encoder layer
    "d_decoder": 320,                       # Decoder Dim
    "num_layers": 1,                        # Deocder Layer
    "num_classes": 29,                      # Output Classes
}


# Instantiate Conformer Model
model = ConformerASR(encoder_params, decoder_params)

# Pass random input to check the model forward pass
x = torch.rand(1, 256, 80)                  # (batch, time_seq, n_feats)
output = model(x)

print(f"Output:\n{output}\n")
print(f"Shape of Output:\n{output.shape}")