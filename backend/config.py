import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAMPLE_RATE: int = 16000
CHUNK: int = 1024
SILENCE_THRESHOLD: float = 50.0
SILENCE_DURATION: float = 2.0  # seconds of silence before auto-stop

UPLOAD_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
STATIC_DIR: str = os.path.join(BASE_DIR, "static")
TOKEN_PATH: str = os.path.join(BASE_DIR, "assets", "tokens.txt")
