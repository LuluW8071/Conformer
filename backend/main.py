"""
Conformer ASR — FastAPI application entry point.

Responsibilities:
  - Parse CLI arguments (model path, token path, host, port)
  - Initialize ASR engine and featurizer during lifespan startup
  - Register all routers
  - Mount static files
  - Launch uvicorn
"""
import argparse
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import SAMPLE_RATE, TOKEN_PATH
from backend.utils.logging import logger
from backend.routes.stream import router as stream_router
from backend.routes.transcribe import router as transcribe_router
from backend.services.decoder import SpeechRecognitionEngine
from backend.services.mel_transform import get_featurizer



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ASR engine before the server accepts requests; clean up on shutdown."""
    model_file = os.environ["ASR_MODEL_FILE"]
    token_path = os.environ.get("ASR_TOKEN_PATH", TOKEN_PATH)

    logger.info("Loading ASR engine from '{}'", model_file)
    app.state.asr_engine = SpeechRecognitionEngine(model_file, token_path)
    app.state.featurizer = get_featurizer(SAMPLE_RATE)
    logger.info("ASR engine ready")

    yield

    logger.info("Shutting down — releasing ASR engine")
    app.state.asr_engine = None
    app.state.featurizer = None


app = FastAPI(
    title="Conformer ASR",
    description="Real-time speech recognition API powered by a Conformer CTC model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(transcribe_router)
app.include_router(stream_router)


# Entrypoint
def main():
    parser = argparse.ArgumentParser(description="Conformer ASR FastAPI backend")
    parser.add_argument("--model_file", required=True, help="Path to the TorchScript model (.pt)")
    parser.add_argument("--token_path", default=TOKEN_PATH, help="Path to tokens.txt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    args = parser.parse_args()

    # Store model config in env so the re-imported module's lifespan can read them
    os.environ["ASR_MODEL_FILE"] = args.model_file
    os.environ["ASR_TOKEN_PATH"] = args.token_path

    logger.info("Starting server on {}:{}", args.host, args.port)
    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
