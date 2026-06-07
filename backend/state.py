"""
Module-level singletons populated during app lifespan startup.
Routes access these via request.app.state (set in main.py lifespan).
"""
from typing import Optional

from backend.services.decoder import SpeechRecognitionEngine

asr_engine: Optional[SpeechRecognitionEngine] = None
featurizer = None
