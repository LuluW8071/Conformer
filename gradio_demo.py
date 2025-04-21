import torch
import torchaudio
import argparse
import gradio as gr

from torchaudio.transforms import Resample
from torchaudio.models.decoder import download_pretrained_files, ctc_decoder

from src.Conformer import get_featurizer

# Constants for decoding
LM_WEIGHT = 1.23
WORD_SCORE = -0.26


def preprocess_audio(audio_file, featurizer, target_sample_rate=16000):
    """
    Preprocess the audio: load, resample, and extract features.
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_file)
        if sample_rate != target_sample_rate:
            waveform = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)
        return featurizer(waveform).permute(0, 2, 1)
    except Exception as e:
        raise ValueError(f"Error in preprocessing audio: {e}")


def decode_emission(emission, tokens, files):
    """
    Decode emissions using a beam search decoder with a language model.
    """
    try:
        beam_search_decoder = ctc_decoder(
            lexicon=files.lexicon,
            tokens=tokens,
            lm=files.lm,
            nbest=1,
            beam_size=50,
            beam_threshold=25,
            beam_size_token=15,
            lm_weight=1.23,
            word_score=-0.26,
        )
        beam_search_result = beam_search_decoder(emission)
        return " ".join(beam_search_result[0][0].words).strip()
    except Exception as e:
        raise ValueError(f"Error in decoding: {e}")


def transcribe(audio_file, model, featurizer, tokens, files):
    """
    Transcribe an audio file using the ASR model and decoder.
    """
    try:
        # Preprocess audio
        waveform = preprocess_audio(audio_file, featurizer)
        
        # Get raw tensor emissions from the model
        emission = model(waveform)
        
        # Decode emissions
        return decode_emission(emission, tokens, files)
    except Exception as e:
        return f"Error processing audio: {e}"


def main(args):
    """
    Main function to launch the Gradio interface.
    """
    # Load ASR Conformer Model and set to eval mode
    model = torch.jit.load(args.model_path)
    model.eval().to('cpu')  # Run on cpu

    # Load tokens and pre-trained language model
    with open(args.token_path, 'r') as f:
        tokens = f.read().splitlines()

    files = download_pretrained_files("librispeech-4-gram")

    # Create feature extractor
    featurizer = get_featurizer()

    # Define Gradio interface
    def gradio_transcribe(audio_file):
        return transcribe(audio_file, model, featurizer, tokens, files)

    interface = gr.Interface(
        fn=gradio_transcribe,
        inputs=gr.Audio(sources="microphone", type="filepath", label="Speak into the microphone"),
        outputs="text",
        title="Conformer-Small ASR Model",
        description=("Speak into the microphone, and the model will transcribe your speech."),
    )

    # Launch the Gradio app
    interface.launch(share=args.share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Model Inference Script")

    parser.add_argument('--model_path', required=True, type=str, help='Path to the model checkpoint file')
    parser.add_argument('--token_path', default="assets/tokens.txt", type=str, help='Path to the tokens file')
    parser.add_argument('--share', action='store_true', help='Share the Gradio app publicly')
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        raise ValueError(f"Fatal error: {e}")
