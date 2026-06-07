import torch
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files

from backend.utils.logging import logger


class SpeechRecognitionEngine:
    """Beam-search CTC decoder backed by a TorchScript Conformer model."""

    def __init__(self, model_file: str, token_path: str) -> None:
        self.model = torch.jit.load(model_file)
        self.model.eval().to("cpu")

        files = download_pretrained_files("librispeech-4-gram")

        with open(token_path) as f:
            tokens = f.read().splitlines()

        self.decoder = ctc_decoder(
            lexicon=files.lexicon,
            tokens=tokens,
            lm=files.lm,
            nbest=1,
            beam_size=150,
            beam_threshold=50,
            beam_size_token=25,
            lm_weight=1.23,
            word_score=-0.26,
        )

        self.greedy_decoder = GreedyCTCDecoder(labels=tokens)
        logger.info("Beam-search decoder with KenLM loaded successfully")


class GreedyCTCDecoder(torch.nn.Module):
    """Greedy (argmax) CTC decoder — fast, no language model."""

    def __init__(self, labels: list[str], blank: int = 0) -> None:
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> list[str]:
        indices = torch.argmax(emission, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = "".join([self.labels[i] for i in indices])
        return joined.replace("-", "").replace("|", " ").strip().split()
