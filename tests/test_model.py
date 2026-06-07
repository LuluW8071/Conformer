import torch

from conformer.utils.logging import logger
from conformer.model import ConformerASR
from conformer.config import ModelConfig

# Load model configuration
config = ModelConfig()
encoder_params, decoder_params = config.encoder.to_dict(), config.decoder.to_dict()

# Instantiate Conformer Model
model = ConformerASR(encoder_params, decoder_params)

logger.info(f"Encoder Params: {encoder_params}")
logger.info(f"Decoder Params: {decoder_params}")

# Pass random input to check the model forward pass
x = torch.rand(1, 256, 80)                  # (batch, time_seq, n_feats)
output = model(x)

logger.info(f"Output: {output}")
logger.info(f"Shape of Output: {output.shape}")