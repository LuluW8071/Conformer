"""
Resample mimic recordings to 16 kHz mono and generate augmented copies
(time-stretch, time-shift, pitch-shift) to add variation to the training set.
"""

import argparse
import os
import logging
import librosa

from tqdm import tqdm

from conformer.scripts.helper import (
    TARGET_SAMPLE_RATE,
    AUGMENTATIONS,
    resample,
    entry,
    write,
    write_audio,
    clean_text,
)

logger = logging.getLogger("Mimic Records Data Prep")

def process_audio(file_path, text, output_dir):
    """Resample one recording, write all augmentations, return their entries."""
    text = clean_text(text)
    base = os.path.splitext(os.path.basename(file_path))[0]

    original = os.path.join(output_dir, f"{base}.flac")
    resample(file_path, original, mono=True)

    samples, sr = librosa.load(original, sr=TARGET_SAMPLE_RATE)
    entries = [entry(original, text)]

    for name, augment in AUGMENTATIONS.items():
        augmented, _ = augment(samples, sr)
        out_path = os.path.join(output_dir, f"{base}_{name}.flac")
        write_audio(out_path, augmented, sr)
        entries.append(entry(out_path, text))

    return entries


def read_transcript(path):
    """Yield (file_name, text) from a `name|text|...` transcript file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 3:
                yield parts[0], parts[1]


def main(args):
    audio_dir = os.path.dirname(args.input_path)
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)

    entries = []
    for file_name, text in tqdm(list(read_transcript(args.input_path))):
        file_path = os.path.join(audio_dir, file_name)
        if os.path.exists(file_path):
            entries.extend(process_audio(file_path, text, output_dir))

    write(entries, args.output_file, indent=2)
    logger.info(f"Dataset saved to {args.output_file} ({len(entries)} entries)")


def parse_args():
    p = argparse.ArgumentParser(
        description="Resample and augment a mimic recording dataset.")
    p.add_argument("input_path", help="path to the transcript file")
    p.add_argument("output_file", help="path to save the JSON manifest")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())