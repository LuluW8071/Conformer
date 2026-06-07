from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

from backend.utils.logging import logger
from backend.utils.helper import transcribe_wav_bytes

router = APIRouter(prefix="/transcribe", tags=["transcribe"])


@router.post("/")
async def transcribe_file(request: Request, file: UploadFile = File(...)):
    """Transcribe an uploaded audio file (WAV, MP3, FLAC, etc.)."""
    asr_engine = request.app.state.asr_engine
    featurizer = request.app.state.featurizer

    if asr_engine is None:
        logger.error("Transcription request received but ASR engine is not initialized")
        return JSONResponse({"error": "ASR engine not initialized"}, status_code=503)

    logger.info("Received file '{}' ({})", file.filename, file.content_type)

    contents = await file.read()
    try:
        transcript = transcribe_wav_bytes(contents, asr_engine, featurizer)
        logger.info("Transcribed '{}' -> {} chars", file.filename, len(transcript))
        return {"transcription": transcript}
    except Exception as exc:
        logger.exception("Transcription failed for '{}'", file.filename)
        return JSONResponse({"error": str(exc)}, status_code=500)
