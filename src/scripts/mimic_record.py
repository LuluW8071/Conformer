"""
Script to resample and add augmented audio samples for variation in training sample using mimic recordings
"""
import os
import json
import argparse
import librosa
import soundfile as sf
import numpy as np
import sox
import re
from tqdm import tqdm


# Augmentation functions
# ==============================
def stretch(audio_data, sr, rate=0.8):
    return librosa.effects.time_stretch(y=audio_data, rate=rate), sr

def shift(audio_data, sr):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1300)
    return np.roll(audio_data, shift_range), sr

def pitch(audio_data, sr, pitch_factor=0.7):
    return librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=pitch_factor), sr
# ==============================


def resample_audio(input_path, output_path):
    """Resample to 16Khz and mono audio format"""
    tfm = sox.Transformer()
    tfm.rate(samplerate=16000)
    tfm.channels(1)
    tfm.build(input_path, output_path)

def clean_text(text):
    """Remove punctuation marks and unwanted symbols"""
    text = re.compile(r'[–\-"`(),:;?!’‘“”…«»\[\]{}&*#@%$^=|_+<>~.ł\t�ß]').sub('', text)
    return text


# Function to process a single file
def process_audio(file_path, text, output_dir):
    text = clean_text(text)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    orig_output = os.path.join(output_dir, f"{base_name}.flac")
    resample_audio(file_path, orig_output)

    audio_data, sr = librosa.load(orig_output, sr=16000)
    output_files = [{"key": orig_output, "text": text}]

    # Apply augmentations
    augmentations = {
        "stretch": stretch,
        "shift": shift,
        "pitch": pitch
    }

    for aug_name, aug_func in augmentations.items():
        aug_data, _ = aug_func(audio_data, sr)
        aug_output = os.path.join(output_dir, f"{base_name}_{aug_name}.flac")
        sf.write(aug_output, aug_data, sr)
        output_files.append({"key": aug_output, "text": text})

    return output_files


def main(input_path, output_file):
    audio_dir = os.path.dirname(input_path)     # Extract base audio directory
    output_dir = os.path.dirname(output_file)   # Extract output directory
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue
            file_name, text, _ = parts
            file_path = os.path.join(audio_dir, file_name)

            if os.path.exists(file_path):
                dataset.extend(process_audio(file_path, text, output_dir))

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)

    print(f"Dataset saved to {output_file}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and augment audio dataset")
    parser.add_argument("input_path", type=str, help="Path to the transcript file")
    parser.add_argument("output_file", type=str, help="Path to save the JSON output")
    
    args = parser.parse_args()
    main(args.input_path, args.output_file)
