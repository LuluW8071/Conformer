"""
Configuration objects for the Conformer ASR training pipeline.
Defaults follow Table 1 (Conformer-S) of https://arxiv.org/pdf/2005.08100
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict


@dataclass
class EncoderConfig:
    d_input: int = 80                          # input features: n-mels
    d_model: int = 144                         # encoder dimension
    num_layers: int = 16                       # number of encoder blocks
    conv_kernel_size: int = 31
    feed_forward_residual_factor: float = 0.5
    feed_forward_expansion_factor: int = 4
    num_heads: int = 4                         # relative multi-head attention heads
    dropout: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecoderConfig:
    d_encoder: int = 144                       # must match EncoderConfig.d_model
    d_decoder: int = 320                       # decoder dimension
    num_layers: int = 1
    num_classes: int = 29                      # output classes (vocab + CTC blank)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelConfig:
    """Top-level model config that ties the encoder and decoder together."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    def __post_init__(self) -> None:
        # The decoder consumes the encoder's output, so `d_encoder` must always
        # equal the encoder's `d_model`. Sync it here so the two values can
        # never silently drift apart when someone changes only one of them.
        self.decoder.d_encoder = self.encoder.d_model

    @property
    def num_classes(self) -> int:
        return self.decoder.num_classes

    @property
    def blank_id(self) -> int:
        # CTC blank is conventionally the last class index (28 for 29 classes).
        return self.decoder.num_classes - 1