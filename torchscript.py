"""Freezes and optimize the trained model checkpoint for inference. Use after training."""

import os 
import argparse
import torch
import torch.nn as nn

from collections import OrderedDict
from Conformer import ConformerEncoder, LSTMDecoder

# Conformer Model Class
class ConformerASR(nn.Module):
    def __init__(self, encoder_params, decoder_params):
        super(ConformerASR, self).__init__()
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


def trace(model):
    """
    Traces the model for optimization.

    Args:
        model (torch.nn.Module): Model to be traced.

    Returns:
        torch.jit.ScriptModule: Traced model.
    """
    model.eval()
    x = torch.rand(1, 300, 80)
    traced = torch.jit.trace(model, (x))
    return traced

def main(args):
    """
    Main function to freeze and optimize the model.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    print("Loading model from", args.model_checkpoint)
    model = ConformerASR(encoder_params, decoder_params)
    checkpoint = torch.load(args.model_checkpoint, weights_only=True, map_location="cpu")
    model_state_dict = checkpoint['state_dict']

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
    model.encoder.load_state_dict(encoder_state_dict)
    model.decoder.load_state_dict(decoder_state_dict)

    print("Tracing model...")
    traced_model = trace(model)
    print("Saving to", args.save_path)
    traced_model.save(os.path.join(args.save_path, 'optimized_model.pt'))
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freeze and optimize the trained model checkpoint for inference.")
    parser.add_argument('--model_checkpoint', type=str, default=None, required=True,
                        help='Checkpoint of model to optimize')
    parser.add_argument('--save_path', type=str, default=None, required=True,
                        help='path to save optimized model')

    args = parser.parse_args()
    
    main(args)