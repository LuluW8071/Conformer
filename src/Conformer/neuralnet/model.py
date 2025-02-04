from torch import nn
from models import ConformerEncoder, LSTMDecoder


class ConformerASR(nn.Module):
    def __init__(self, encoder_params, decoder_params):
        super(ConformerASR, self).__init__()
        self.encoder = ConformerEncoder(**encoder_params)
        self.decoder = LSTMDecoder(**decoder_params)

    def forward(self, x, mask=None):
        encoder_output = self.encoder(x, mask)
        decoder_output = self.decoder(encoder_output)
        return decoder_output