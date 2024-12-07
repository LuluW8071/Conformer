import argparse
import os

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from dataset import get_featurizer
from decoder import SpeechRecognitionEngine
from engine import Recorder

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize shared objects at startup
recorder = Recorder()               # Initialize recorder
featurizer = get_featurizer(16000)  # Initialize featurizer

asr_engine = None  # Initialize to None, to avoid issues during startup

# Serve static files (index.html)
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(STATIC_FOLDER, "index.html")


@app.route("/transcribe/", methods=["POST"])
def transcribe_audio():
    """
    Transcribe an uploaded audio file using the preloaded ASR engine.
    """
    try:
        if asr_engine is None:
            return jsonify({"error": "ASR Engine is not initialized."}), 500

        # Check if file is in request
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Secure filename and save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        print(f"File saved: {file_path}")

        recorded_audio = recorder.record()
        recorder.save(recorded_audio, "audio_temp.wav")
        print("\nAudio recorded")

        # Use the preloaded ASR Engine to transcribe
        transcript = asr_engine.transcribe(asr_engine.model, featurizer, "audio_temp.wav")

        print("\nTranscription:")
        print(transcript)

        return jsonify({"transcription": transcript})

    except Exception as e:
        return jsonify({"error": f"Internal server error: {e}"}), 500


def main(args):
    global asr_engine
    print("Loading Speech Recognition Engine into cache...")
    try:
        asr_engine = SpeechRecognitionEngine(args.model_file, args.token_path) 
        print("ASR Engine loaded successfully.")
    except Exception as e:
        print(f"Error loading ASR Engine: {e}")
        asr_engine = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Demo: Record and Transcribe Audio")
    parser.add_argument('--model_file', type=str, required=True, help='Path to the optimized ASR model.')
    parser.add_argument('--token_path', type=str, default="assets/tokens.txt", help='Path to the tokens file.')
    args = parser.parse_args()

    main(args)
    
    app.run(host="127.0.0.1", port=8080, debug=True)
