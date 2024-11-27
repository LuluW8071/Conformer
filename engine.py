import pyaudio
import wave
import math
import time
import struct
import argparse
import torch
import torchaudio
from dataset import get_featurizer
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files


class Recorder:
    @staticmethod
    def rms(frame):
        """
        Calculate the Root Mean Square (RMS) value of a frame for silence detection.
        """
        count = len(frame) // 2
        format = f"{count}h"
        shorts = struct.unpack(format, frame)

        sum_squares = sum((sample * (1.0 / 32768.0)) ** 2 for sample in shorts)
        rms = math.sqrt(sum_squares / count) * 1000
        return rms

    def __init__(self, sample_rate=16000, chunk=1024, silence_threshold=50, silence_duration=5):
        self.chunk = chunk
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.p = pyaudio.PyAudio()
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk
            )
        except Exception as e:
            print(f"Error initializing audio stream: {e}")
            self.stream = None

    def record(self):
        """
        Record audio until silence is detected for a certain duration.
        """
        if not self.stream:
            raise RuntimeError("Audio stream is not initialized.")

        print("Recording... Speak into the microphone.")
        audio_frames = []
        silence_start = None

        while True:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            audio_frames.append(data)

            # Detect silence
            if self.rms(data) < self.silence_threshold:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= self.silence_duration:
                    print("Silence detected. Stopping recording.")
                    break
            else:
                silence_start = None

        return audio_frames

    def save(self, waveforms, filename="audio_temp.wav"):
        """
        Save the recorded audio to a WAV file.
        """
        try:
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b"".join(waveforms))
            print(f"Audio saved to {filename}")
            return filename
        except Exception as e:
            raise RuntimeError(f"Error saving audio: {e}")


class SpeechRecognitionEngine:
    """
    ASR engine to transcribe recorded audio.
    """
    def __init__(self, model_file, token_path, ken_lm_file=None):
        self.model = torch.jit.load(model_file)
        self.model.eval().to('cpu')  # Ensure the model runs on CPU
        self.featurizer = get_featurizer(16000)

        # Load decoder files and tokens
        files = download_pretrained_files("librispeech-4-gram")
        # print(f"Loaded decoder files: {files}")
        with open(token_path, 'r') as f:
            tokens = f.read().splitlines()
            
        lm = files.lm if ken_lm_file is None else ken_lm_file

        self.decoder = ctc_decoder(
            lexicon=files.lexicon,
            tokens=tokens,
            lm=lm,
            nbest=10,
            beam_size=100,
            lm_weight=3.23,
            word_score=-0.26,
        )

    def transcribe(self, filename):
        """
        Transcribe audio from a file using the ASR model.
        """
        try:
            waveform, _ = torchaudio.load(filename)
            mel = self.featurizer(waveform).permute(0, 2, 1)  # Prepare mel features
            with torch.inference_mode():
                out = self.model(mel)
                results = self.decoder(out)
                return " ".join(results[0][0].words).strip()
        except Exception as e:
            raise RuntimeError(f"Error during transcription: {e}")


def main(args):
    try:
        # Initialize recorder and ASR engine
        recorder = Recorder()
        asr_engine = SpeechRecognitionEngine(args.model_file, args.token_path, args.ken_lm_file)

        # Record audio
        recorded_audio = recorder.record()
        audio_file = recorder.save(recorded_audio, "audio_temp.wav")

        # Transcribe audio
        transcript = asr_engine.transcribe(audio_file)
        print("Transcription:")
        print(transcript)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Demo: Record and Transcribe Audio")
    parser.add_argument('--model_file', type=str, required=True, help='Path to the optimized ASR model.')
    parser.add_argument('--token_path', type=str, default="assets/tokens.txt", help='Path to the tokens file.')
    parser.add_argument('--ken_lm_file', type=str, default=None, help='Path to an optional KenLM file.')
    args = parser.parse_args()

    main(args)
