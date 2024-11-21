import torch
from torch import nn
from collections import OrderedDict

from Conformer import ConformerEncoder, LSTMDecoder


class SpeechRecognition(nn.Module):
    def __init__(self, encoder_params, decoder_params):
        super(SpeechRecognition, self).__init__()
        self.encoder = ConformerEncoder(**encoder_params)
        self.decoder = LSTMDecoder(**decoder_params)

    def forward(self, x):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return decoder_output

# Hyper parameters of trained_model (conformer small)
encoder_params = {
    'd_input': 80,       # Input features: n-mels
    'd_model': 144,      # Encoder Dims
    'num_layers': 16,    # Encoder Layers
    'conv_kernel_size': 31,
    'feed_forward_residual_factor': 0.5,
    'feed_forward_expansion_factor': 4,
    'num_heads': 4,      # Relative MultiHead Attetion Heads
    'dropout': 0.1,
}

decoder_params = {
    'd_encoder': 144,    # Match with Encoder layer
    'd_decoder': 320,    # Decoder Dim
    'num_layers': 1,     # Deocder Layer
    'num_classes': 29,   # Output Classes
}



def load_model(checkpoint_path):
    """ Load the checkpoint weights """
    model = SpeechRecognition(encoder_params, decoder_params)
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    model_state_dict = checkpoint.get('state_dict', checkpoint)

    # Initialize state dictionaries
    encoder_state_dict = OrderedDict()
    decoder_state_dict = OrderedDict()

    # Separate encoder and decoder state dictionaries
    for k, v in model_state_dict.items():
        if k.startswith('model._orig_mod.encoder.'):
            name = k.replace('model._orig_mod.encoder.', '')
            encoder_state_dict[name] = v
        elif k.startswith('model._orig_mod.decoder.'):
            name = k.replace('model._orig_mod.decoder.', '')
            decoder_state_dict[name] = v

    # Load state dictionaries into the model
    model.encoder.load_state_dict(encoder_state_dict, strict=False)
    model.decoder.load_state_dict(decoder_state_dict, strict=False)

    return model
