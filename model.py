from torch import nn
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