import os
import argparse
import torchaudio
import torch
from torchaudio.datasets import LIBRISPEECH
from jiwer import wer, cer
from tqdm import tqdm

from decoder import SpeechRecognitionEngine


def load_featurizer():
    """
    Load the mel-spectrogram featurizer consistent with the model's training.
    """
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=80,
    )


def evaluate(model_path, token_path, dataset_path, split="test-clean", max_samples=None):
    print("🔧 Initializing ASR Engine...")
    featurizer = load_featurizer()
    engine = SpeechRecognitionEngine(model_path, token_path)

    print(f"📚 Loading LibriSpeech split: {split}")
    dataset = LIBRISPEECH(dataset_path, url=split, download=True)

    total_wer, total_cer = 0.0, 0.0
    sample_count = 0

    iterable = enumerate(dataset)
    if max_samples:
        iterable = zip(range(max_samples), dataset)

    print("🚀 Transcribing and evaluating...")
    for idx, (waveform, sample_rate, reference, *_ ) in tqdm(iterable, total=max_samples or len(dataset), desc="Processing"):
        try:
            tmp_wav_path = f"temp_sample_{idx}.wav"
            torchaudio.save(tmp_wav_path, waveform, sample_rate)

            prediction = engine.transcribe(engine.model, featurizer, tmp_wav_path)
            os.remove(tmp_wav_path)

            reference = reference.lower().strip()
            prediction = prediction.lower().strip()

            sample_wer = wer(reference, prediction)
            sample_cer = cer(reference, prediction)
            total_wer += sample_wer
            total_cer += sample_cer
            sample_count += 1

        except Exception as e:
            tqdm.write(f"[{idx}] ⚠️ Error during transcription: {e}")
            continue

    if sample_count == 0:
        print("❌ No samples processed. Evaluation aborted.")
    else:
        avg_wer = total_wer / sample_count
        avg_cer = total_cer / sample_count
        print("\n✅ Evaluation Complete!")
        print(f"Total Samples: {sample_count}")
        print(f"Average WER: {avg_wer:.4f}")
        print(f"Average CER: {avg_cer:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a TorchScript ASR model on LibriSpeech dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the TorchScript model file")
    parser.add_argument("--token_path", type=str, required=True, help="Path to the tokens.txt file")
    parser.add_argument("--dataset_path", type=str, required=True, help="Root path to LibriSpeech dataset")
    parser.add_argument("--split", type=str, default="test-clean", choices=["test-clean", "test-other"], help="LibriSpeech split to evaluate")
    parser.add_argument("--max_samples", type=int, default=None, help="Number of samples to evaluate (None for full set)")

    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        token_path=args.token_path,
        dataset_path=args.dataset_path,
        split=args.split,
        max_samples=args.max_samples
    )