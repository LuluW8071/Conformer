import torch
import torchaudio

from torchaudio.models.decoder import ctc_decoder, download_pretrained_files


class SpeechRecognitionEngine:
    """
    ASR engine to transcribe recorded audio.
    """
    def __init__(self, model_file, token_path):
        self.model = torch.jit.load(model_file)
        self.model.eval().to('cpu')

        # Load decoder files and tokens
        files = download_pretrained_files("librispeech-4-gram")

        with open(token_path, 'r') as f:
            tokens = f.read().splitlines()

        self.decoder = ctc_decoder(
            lexicon=files.lexicon,
            tokens=tokens,
            lm=files.lm,
            nbest=1,
            beam_size=500,
            beam_threshold=100,
            beam_size_token = 30,
            lm_weight=1.23,
            word_score=-0.26,
        )

        print("Loaded beam search with Ken LM")


    def transcribe(self, model, featurizer, filename):
        """
        Transcribe audio from a file using the ASR model.
        """

        try:
            waveform, _ = torchaudio.load(filename)
            mel = featurizer(waveform).permute(0, 2, 1)  # Prepare mel features
            with torch.inference_mode():
                out = model(mel)
                results = self.decoder(out)
                return " ".join(results[0][0].words).strip()
        except Exception as e:
            raise RuntimeError(f"Error during transcription: {e}")


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Decode the emission tensor to a string."""
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]

        # Join the decoded indices into a string and replace "-" with no space and "|" with spaces
        joined = "".join([self.labels[i] for i in indices])
        return joined.replace("-", "").replace("|", " ").strip().split()

