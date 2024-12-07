import pyaudio
import wave
import math
import time
import struct
import argparse
from dataset import get_featurizer
from decoder import SpeechRecognitionEngine


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

    def __init__(self, sample_rate=16000, chunk=1024, silence_threshold=50, silence_duration=3):
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

        print("\nRecording... Speak into the microphone.")
        audio_frames = []
        silence_start = None

        while True:
            print(f"Recording now...")
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


def main(args):
    try:
        # Initialize recorder and ASR engine
        recorder = Recorder()
        asr_engine = SpeechRecognitionEngine(args.model_file, args.token_path)
        featurizer = get_featurizer(16000)

        # Record audio
        recorded_audio = recorder.record()
        audio_file = recorder.save(recorded_audio, "audio_temp.wav")

        # Transcribe audio
        transcript = asr_engine.transcribe(asr_engine.model, featurizer, audio_file)
        print("\nTranscription:")
        print(transcript)


    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Demo: Record and Transcribe Audio")
    parser.add_argument('--model_file', type=str, required=True, help='Path to the optimized ASR model.')
    parser.add_argument('--token_path', type=str, default="assets/tokens.txt", help='Path to the tokens file.')
    args = parser.parse_args()

    main(args)
