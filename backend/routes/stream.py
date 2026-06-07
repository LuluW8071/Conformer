import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.config import SILENCE_DURATION, SILENCE_THRESHOLD
from backend.utils.logging import logger
from backend.utils.helper import compute_rms, frames_to_wav_bytes, transcribe_wav_bytes

router = APIRouter(prefix="/ws", tags=["stream"])


@router.websocket("/stream/")
async def stream_mic(websocket: WebSocket):
    """
    Record audio until silence or manual stop, then transcribe once.

    Client → server  : binary PCM Int16 chunks (mono, 16 kHz) | text "stop"
    Server → client  : "listening" | "silence"
                       {"transcription": "..."} | {"error": "..."}
    """
    await websocket.accept()
    client = websocket.client
    logger.info("WebSocket connected from {}:{}", client.host, client.port)

    asr_engine = websocket.app.state.asr_engine
    featurizer = websocket.app.state.featurizer

    if asr_engine is None:
        await websocket.send_json({"error": "ASR engine not initialized"})
        await websocket.close(code=1013)
        return

    await websocket.send_text("listening")

    audio_frames: list[bytes] = []
    silence_start: float | None = None

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message:
                chunk: bytes = message["bytes"]
                audio_frames.append(chunk)

                rms_val = compute_rms(chunk)
                if rms_val < SILENCE_THRESHOLD:
                    if silence_start is None:
                        silence_start = time.monotonic()
                    elif time.monotonic() - silence_start >= SILENCE_DURATION:
                        logger.debug("Silence detected — stopping stream")
                        await websocket.send_text("silence")
                        break
                else:
                    silence_start = None

            elif "text" in message and message["text"].strip().lower() == "stop":
                logger.debug("Client sent stop signal")
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected from {}:{}", client.host, client.port)
        return

    if not audio_frames:
        await websocket.send_json({"error": "No audio received"})
        return

    try:
        wav_bytes = frames_to_wav_bytes(audio_frames)
        transcript = transcribe_wav_bytes(wav_bytes, asr_engine, featurizer)
        logger.info("Transcription complete ({} frames) -> {} chars", len(audio_frames), len(transcript))
        await websocket.send_json({"transcription": transcript})
    except Exception as exc:
        logger.exception("Transcription failed")
        await websocket.send_json({"error": str(exc)})

    await websocket.close()
