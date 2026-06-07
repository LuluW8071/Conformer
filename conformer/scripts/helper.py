"""Text normalization, Audio resampling and augmentation helpers shared across prep scripts."""

import os
import re
import json
import random

import librosa
import numpy as np
import soundfile as sf
import sox


from typing import Iterable, List, Tuple
 

# Punctuation / symbols stripped from transcripts before training.
_PUNCTUATION = re.compile(r'[–\-"`(),:;?!’‘“”…«»\[\]{}&*#@%$^=|_+<>~.ł\t�ß]')
TARGET_SAMPLE_RATE = 16000
Entry = dict

def clean_text(text: str) -> str:
    """Remove punctuation marks and unwanted symbols from a transcript."""
    return _PUNCTUATION.sub('', text)


def resample(
    input_path: str, 
    output_path: str,
    sample_rate: int = TARGET_SAMPLE_RATE, 
    mono: bool = True
) -> None:
    """Resample an audio file to a target sample rate."""
    tfm = sox.Transformer()
    tfm.rate(samplerate=sample_rate)
    if mono:
        tfm.channels(1)
    tfm.build(input_filepath=input_path, output_filepath=output_path)


def write_audio(path: str, audio: np.ndarray, sr: int) -> None:
    """Write a numpy waveform to disk."""
    sf.write(path, audio, sr)


# Audio Augmentations 
def time_stretch(audio, sr, rate: float = 0.8):
    return librosa.effects.time_stretch(y=audio, rate=rate), sr


def time_shift(audio, sr, max_seconds: float = 5.0, scale: int = 1300):
    shift_amount = int(np.random.uniform(low=-max_seconds, high=max_seconds) * scale)
    return np.roll(audio, shift_amount), sr


def pitch_shift(audio, sr, n_steps: float = 0.7):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps), sr

 
 
def entry(key: str, text: str) -> Entry:
    """Build a single manifest record."""
    return {"key": key, "text": text}
 
 
def write(
    entries: Iterable[Entry], 
    path: str, 
    indent: int = 4
) -> None:
    """Write manifest entries to a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(entries), f, ensure_ascii=False, indent=indent)
 
 
def split_train_test(
    entries: Iterable[Entry], 
    test_percent: float,
    shuffle: bool = True, 
    seed: int = None
) -> Tuple[List[Entry], List[Entry]]:
    """Split entries into (train, test)."""
    entries = list(entries)
    if shuffle:
        random.Random(seed).shuffle(entries)
    cutoff = int(len(entries) * (1 - test_percent / 100))
    return entries[:cutoff], entries[cutoff:]


AUGMENTATIONS = {
    "stretch": time_stretch,
    "shift": time_shift,
    "pitch": pitch_shift,
}