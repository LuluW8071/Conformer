"""
Audio utilities: RMS silence detection, PCM→WAV conversion, and transcription.
All functions are pure (no global state); engine/featurizer are injected via args.
"""
import io
import math
import struct
import wave

import torch
import torchaudio

from backend.config import SAMPLE_RATE


def compute_rms(frame: bytes) -> float:
    """Return the RMS energy (×1000) of a raw PCM Int16 frame."""
    count = len(frame) // 2
    shorts = struct.unpack(f"{count}h", frame)
    sum_sq = sum((s / 32768.0) ** 2 for s in shorts)
    return math.sqrt(sum_sq / count) * 1000


def frames_to_wav_bytes(frames: list[bytes], sample_rate: int = SAMPLE_RATE) -> bytes:
    """Pack a list of raw PCM Int16 frames into an in-memory WAV buffer."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # paInt16
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


def transcribe_wav_bytes_greedy(wav_bytes: bytes, asr_engine, featurizer) -> str:
    """Fast greedy CTC decode — used for real-time partial results."""
    buf = io.BytesIO(wav_bytes)
    waveform, sr = torchaudio.load(buf)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    mel = featurizer(waveform).permute(0, 2, 1)
    with torch.inference_mode():
        out = asr_engine.model(mel)
        words = asr_engine.greedy_decoder(out.squeeze(0))
        return " ".join(words).strip()


def transcribe_wav_bytes(wav_bytes: bytes, asr_engine, featurizer) -> str:
    """
    Transcribe raw WAV bytes with the given engine and mel featurizer.
    Resamples automatically when the embedded sample-rate differs from SAMPLE_RATE.
    """
    buf = io.BytesIO(wav_bytes)
    waveform, sr = torchaudio.load(buf)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    mel = featurizer(waveform).permute(0, 2, 1)
    with torch.inference_mode():
        out = asr_engine.model(mel)
        results = asr_engine.decoder(out)
        return " ".join(results[0][0].words).strip()
