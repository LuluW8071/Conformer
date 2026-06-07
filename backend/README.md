# Conformer ASR — FastAPI Backend

Real-time and file-based speech recognition API powered by a Conformer CTC model with KenLM beam-search decoding.

## Project Structure

```
backend/
├── main.py               # App factory, lifespan startup, router registration
├── config.py             # All constants (sample rate, silence thresholds, paths)
├── logger.py             # Shared structured logger
├── state.py              # Type stubs for app.state (asr_engine, featurizer)
├── routes/
│   ├── transcribe.py     # POST /transcribe/
│   └── stream.py         # WS  /ws/stream/
├── services/
│   └── decoder.py        # SpeechRecognitionEngine + GreedyCTCDecoder
├── utils/
│   └── helper.py         # compute_rms, frames_to_wav_bytes, transcribe_wav_bytes
└── uploads/              # Temporary upload directory (auto-created)
```

## Requirements

Install dependencies from the project root:

```bash
uv pip install -r requirements.txt
# or
pip install -r requirements.txt
```

Key backend dependencies: `fastapi`, `uvicorn[standard]`, `python-multipart`, `torch`, `torchaudio`.

## Running the Server

```bash
python -m backend.main \
  --model_file path/to/model.pt \
  --token_path assets/tokens.txt \
  --host 127.0.0.1 \
  --port 8000
```

Add `--reload` for development auto-reload (do not use in production).

## API Reference

### `POST /transcribe/`

Upload an audio file for transcription.

| Field  | Type | Description                     |
|--------|------|---------------------------------|
| `file` | form | Audio file (WAV, MP3, FLAC …)  |

**Response:**
```json
{ "transcription": "hello world" }
```

---

### `WS /ws/stream/`

WebSocket endpoint for real-time microphone streaming.

**Client → Server:**
| Message type | Content | Meaning |
|---|---|---|
| binary | Raw PCM Int16 bytes (mono, 16 kHz) | Audio chunk |
| text | `"stop"` | Manual end-of-speech signal |

**Server → Client:**
| Message | Meaning |
|---|---|
| `"listening"` | Connection accepted, ready to receive audio |
| `"silence"` | Silence detected for ≥ `SILENCE_DURATION` seconds; stream stopped |
| `{"transcription": "..."}` | Final transcription result |
| `{"error": "..."}` | Failure details |

**Example JavaScript client:**

```js
const ws = new WebSocket("ws://localhost:8000/ws/stream/");
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const recorder = new MediaRecorder(stream);

ws.onmessage = ({ data }) => {
  if (data === "listening") recorder.start(100);  // send 100 ms chunks
  else if (data === "silence") recorder.stop();
  else console.log(JSON.parse(data));             // { transcription: "..." }
};

recorder.ondataavailable = ({ data }) => ws.send(data);
```

## Configuration

Edit `backend/config.py` to adjust:

| Constant | Default | Description |
|---|---|---|
| `SAMPLE_RATE` | `16000` | Audio sample rate (Hz) |
| `SILENCE_THRESHOLD` | `50.0` | RMS value below which audio is considered silent |
| `SILENCE_DURATION` | `2.0` | Seconds of silence before auto-stop |
| `CHUNK` | `1024` | PCM chunk size (samples) |
