# app.py — Render-compatible, free, fully local transcription + summarization + QA
from flask import Flask, render_template, request, jsonify, send_file
from transformers import pipeline
import os
import uuid
from nltk.tokenize import sent_tokenize
import nltk
import logging
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import PyPDF2
from vosk import Model, KaldiRecognizer
import wave
import json
from pytube import YouTube
import traceback
import shutil

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure NLTK punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Flask app
app = Flask(__name__, static_folder='static')

# Temporary folder for Render
TEMP_DIR = "/tmp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Lazy-loaded models
_summarizer = None
_qa_generator = None
_vosk_model = None
VOSK_MODEL_PATH = os.path.join(TEMP_DIR, "vosk-model-small-en-us-0.15")  # adjust path

# ------------------- Helpers -------------------

def load_transformer_models():
    global _summarizer, _qa_generator
    if _summarizer is None:
        logger.info("Loading summarizer model...")
        _summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", tokenizer="sshleifer/distilbart-cnn-12-6")
        logger.info("Summarizer loaded.")
    if _qa_generator is None:
        logger.info("Loading question-generation model...")
        _qa_generator = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-question-generation-ap")
        logger.info("QA generator loaded.")

def load_vosk_model():
    global _vosk_model
    if _vosk_model is None:
        if not os.path.exists(VOSK_MODEL_PATH):
            raise FileNotFoundError(f"Vosk model not found at {VOSK_MODEL_PATH}. Upload or download it first.")
        _vosk_model = Model(VOSK_MODEL_PATH)
        logger.info("Vosk model loaded.")
    return _vosk_model

def preprocess_text(text):
    try:
        sentences = sent_tokenize(text)
        cleaned_sentences = [s.strip() for s in sentences if len(s.split()) > 3]
        return " ".join(cleaned_sentences)
    except Exception as e:
        logger.error("Error in preprocess_text: %s", e)
        return text

def summarize_text(text):
    try:
        load_transformer_models()
        text = preprocess_text(text)
        # Chunk to avoid model limit
        max_chars = 1000
        chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
        summaries = []
        for chunk in chunks:
            max_len = max(50, int(len(chunk.split()) * 0.8))
            summary = _summarizer(chunk, max_length=max_len, min_length=30, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        return " ".join(summaries)
    except Exception as e:
        logger.error("Error in summarize_text: %s", e)
        traceback.print_exc()
        return "Error during summarization."

def generate_questions(summary):
    try:
        load_transformer_models()
        sentences = sent_tokenize(summary)
        questions = []
        for sentence in sentences[:5]:
            q = _qa_generator(f"Generate a question based on: {sentence}", max_length=64, num_return_sequences=1, do_sample=True)[0]['generated_text']
            questions.append(q)
        return questions
    except Exception as e:
        logger.error("Error in generate_questions: %s", e)
        traceback.print_exc()
        return ["Error generating questions."]

def extract_text_from_pdf(filepath):
    try:
        reader = PyPDF2.PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        logger.error("PDF extraction error: %s", e)
        return None

def extract_video_id(url):
    try:
        parsed_url = urlparse(url)
        if "youtu.be" in parsed_url.netloc:
            return parsed_url.path.lstrip('/')
        elif "youtube.com" in parsed_url.netloc:
            return parse_qs(parsed_url.query).get('v', [None])[0]
        return None
    except Exception as e:
        logger.error("Error extracting video ID: %s", e)
        return None

def download_youtube_audio(url):
    """Download audio from YouTube to TEMP_DIR"""
    try:
        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        filename = f"{uuid.uuid4()}.mp4"
        filepath = os.path.join(TEMP_DIR, filename)
        audio_stream.download(output_path=TEMP_DIR, filename=filename)
        return filepath
    except Exception as e:
        logger.error("Error downloading YouTube audio: %s", e)
        return None

def transcribe_audio_vosk(audio_path):
    try:
        model = load_vosk_model()
        wf = wave.open(audio_path, "rb")
        rec = KaldiRecognizer(model, wf.getframerate())
        transcript = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                transcript += " " + res.get("text", "")
        res = json.loads(rec.FinalResult())
        transcript += " " + res.get("text", "")
        return transcript.strip()
    except Exception as e:
        logger.error("Vosk transcription error: %s", e)
        traceback.print_exc()
        return None

def get_youtube_transcript(url):
    """Try YouTube captions first, fallback to audio download + Vosk transcription"""
    try:
        video_id = extract_video_id(url)
        if not video_id:
            return None
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join([item['text'] for item in transcript_list])
            logger.info("Used YouTube captions.")
            return text
        except Exception:
            logger.info("Captions not found, downloading audio for local transcription...")
            audio_file = download_youtube_audio(url)
            if audio_file:
                text = transcribe_audio_vosk(audio_file)
                os.remove(audio_file)
                return text
            return None
    except Exception as e:
        logger.error("Error in get_youtube_transcript: %s", e)
        return None

def save_to_file(content, filename):
    path = os.path.join(TEMP_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

# ------------------- Flask Routes -------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        file = request.files.get('file')
        youtube_url = request.form.get('youtube_url')
        transcript = None

        if not file and not youtube_url:
            return jsonify({"error": "No file or URL provided"}), 400

        # ---------------- File Upload ----------------
        if file:
            filename = f"{uuid.uuid4()}_{file.filename}"
            filepath = os.path.join(TEMP_DIR, filename)
            file.save(filepath)
            if filename.lower().endswith(".pdf"):
                transcript = extract_text_from_pdf(filepath)
            else:
                transcript = transcribe_audio_vosk(filepath)
            os.remove(filepath)

        # ---------------- YouTube URL ----------------
        elif youtube_url:
            transcript = get_youtube_transcript(youtube_url)

        if not transcript:
            return jsonify({"error": "Transcription failed"}), 500

        # ---------------- Summarization ----------------
        summary = summarize_text(transcript)
        # ---------------- Question Generation ----------------
        questions = generate_questions(summary)

        # Save files
        transcript_file = save_to_file(transcript, "transcript.txt")
        summary_file = save_to_file(summary, "summary.txt")
        questions_file = save_to_file("\n".join(questions), "questions.txt")

        return jsonify({
            "transcript_content": transcript,
            "summary_content": summary,
            "qa_content": questions,
            "transcript_file": f"/download/{os.path.basename(transcript_file)}",
            "summary_file": f"/download/{os.path.basename(summary_file)}",
            "qa_file": f"/download/{os.path.basename(questions_file)}"
        })

    except Exception as e:
        logger.error("Error in /process: %s", e)
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    filepath = os.path.join(TEMP_DIR, filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({"error": "File not found"}), 404

@app.route('/remove-file', methods=['POST'])
def remove_file():
    try:
        data = request.get_json()
        if not data or 'filename' not in data:
            return jsonify({"error": "No filename provided"}), 400
        filepath = os.path.join(TEMP_DIR, data['filename'])
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({"success": True})
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        logger.error("Error in /remove-file: %s", e)
        return jsonify({"error": "Internal error"}), 500

# ------------------- Run App -------------------

if __name__ == '__main__':
    # Local testing only; Render will run via Gunicorn
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=False)
