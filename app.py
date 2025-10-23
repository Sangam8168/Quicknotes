# Updated app.py — use local models (faster-whisper + transformers) and ready for Docker/Gunicorn
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
from faster_whisper import WhisperModel
import time
import traceback

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure NLTK punkt tokenizer is available (download only if missing)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = Flask(__name__, static_folder='static')

# Lazy-loaded models (so startup is fast)
_summarizer = None
_qa_generator = None
_whisper_model = None

def load_transformer_models(summarizer_model_name="facebook/bart-large-cnn", qa_model_name="google/flan-t5-base"):
    global _summarizer, _qa_generator
    if _summarizer is None:
        logger.info("Loading summarizer model...")
        _summarizer = pipeline("summarization", model=summarizer_model_name, tokenizer=summarizer_model_name)
        logger.info("Summarizer loaded.")
    if _qa_generator is None:
        logger.info("Loading question-generation model...")
        _qa_generator = pipeline("text2text-generation", model=qa_model_name)
        logger.info("QA generator loaded.")

def load_whisper_model(model_size="small", device="cpu", compute_type="int8"):
    global _whisper_model
    if _whisper_model is None:
        logger.info(f"Loading Whisper model (size={model_size}, device={device}, compute_type={compute_type})...")
        _whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("Whisper model loaded.")
    return _whisper_model

def extract_video_id(url):
    """Extract YouTube video ID from various URL formats."""
    try:
        parsed_url = urlparse(url)
        if "youtu.be" in parsed_url.netloc:  # Shortened URL
            return parsed_url.path.lstrip('/')
        elif "youtube.com" in parsed_url.netloc:  # Full URL
            query_params = parse_qs(parsed_url.query)
            return query_params.get('v', [None])[0]
        return None
    except Exception as e:
        logger.error(f"Error extracting video ID: {e}")
        return None

def get_youtube_transcript(url):
    """Get transcript from YouTube video using youtube_transcript_api (no API key required)."""
    try:
        video_id = extract_video_id(url)
        if not video_id:
            raise ValueError("Could not extract video ID from URL")
        logger.info(f"Getting transcript for video ID: {video_id}")
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = " ".join([item['text'] for item in transcript_list])
        return full_transcript
    except Exception as e:
        logger.error(f"YouTube transcript error: {e}")
        return None

def extract_text_from_pdf(filepath):
    """Extracts all text from a PDF file."""
    try:
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return None

def transcribe_audio(filepath, model_size="small", device="cpu", compute_type="int8"):
    """
    Transcribe an audio/video file using a local Whisper model (faster-whisper).
    This avoids any paid external transcription API.
    Requirements: faster-whisper and ffmpeg available on the system.
    """
    try:
        model = load_whisper_model(model_size=model_size, device=device, compute_type=compute_type)
        segments, info = model.transcribe(filepath, beam_size=5)
        text = " ".join([segment.text.strip() for segment in segments if segment.text.strip()])
        logger.info(f"Transcription completed (duration={getattr(info,'duration', 'unknown')}s)")
        return text
    except Exception as e:
        logger.error(f"Local transcription error: {e}")
        traceback.print_exc()
        return None

def preprocess_text(text):
    logger.debug("Preprocessing transcript...")
    try:
        sentences = sent_tokenize(text)
        cleaned_sentences = [sentence.strip() for sentence in sentences if len(sentence.split()) > 3]
        preprocessed_text = " ".join(cleaned_sentences)
        logger.debug("Preprocessed text length: %d", len(preprocessed_text))
        return preprocessed_text
    except Exception as e:
        logger.error("Error during preprocessing: %s", e)
        return text

def summarize_text(text):
    logger.info("Summarizing...")
    try:
        # Lazy-load transformer models
        load_transformer_models()
        preprocessed_text = preprocess_text(text)
        # Simple character-based chunking (conservative)
        max_chars = 1000
        chunks = [preprocessed_text[i:i + max_chars] for i in range(0, len(preprocessed_text), max_chars)]
        summaries = []
        for chunk in chunks:
            input_length = len(chunk.split())  # Count words
            max_len = max(50, int(input_length * 0.8))
            summary_result = _summarizer(chunk, max_length=max_len, min_length=30, do_sample=False)
            summary = summary_result[0]['summary_text']
            summaries.append(summary)
        summarized_text = " ".join(summaries)
        return summarized_text
    except Exception as e:
        logger.error("Error during summarization: %s", e)
        traceback.print_exc()
        return "Error during summarization."

def generate_questions_from_summary(summary):
    logger.info("Generating questions from summary...")
    logger.debug("Summary input length: %d", len(summary) if summary else 0)
    try:
        load_transformer_models()
        sentences = sent_tokenize(summary)
        if not sentences:
            logger.warning("No sentences found in summary.")
            return ["No questions generated: summary was empty."]
        questions = []
        for sentence in sentences[:5]:
            qa = _qa_generator(
                f"Generate a question based on: {sentence}",
                max_length=64,
                num_return_sequences=1,
                do_sample=True
            )[0]['generated_text']
            questions.append(qa)
        return questions
    except Exception as e:
        logger.error("Error generating questions: %s", e)
        traceback.print_exc()
        return [f"Error generating questions: {str(e)}"]

def save_to_file(content, filename):
    """Save the given content to a text file."""
    filepath = os.path.join("temp", filename)
    os.makedirs("temp", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(content)
    return filepath

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/remove-file', methods=['POST'])
def remove_file():
    try:
        form_data = request.get_json()
        if not form_data or 'filename' not in form_data:
            return jsonify({"error": "No filename provided"}), 400
        filename = form_data['filename']
        filepath = os.path.join("temp", filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({"success": True})
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        logger.error("Error in remove_file: %s", e)
        return jsonify({"error": "Internal error"}), 500

@app.route('/process', methods=['POST'])
def process():
    try:
        start_time = time.time()
        file = request.files.get('file')
        youtube_url = request.form.get('youtube_url')
        transcript = None
        step_times = {}

        if not file and not youtube_url:
            return jsonify({"error": "No file or URL provided"}), 400

        # Handle file upload
        if file:
            filename = f"{uuid.uuid4()}_{file.filename}"
            filepath = os.path.join("temp", filename)
            os.makedirs("temp", exist_ok=True)
            file.save(filepath)
            if filename.lower().endswith('.pdf'):
                t0 = time.time()
                transcript = extract_text_from_pdf(filepath)
                step_times['pdf_extract'] = time.time() - t0
                if not transcript:
                    return jsonify({"error": "Failed to extract text from PDF"}), 400
            else:
                # Transcribe locally using faster-whisper (no API key required)
                t0 = time.time()
                transcript = transcribe_audio(filepath)
                step_times['audio_transcribe'] = time.time() - t0

        # Handle YouTube URL
        elif youtube_url:
            t0 = time.time()
            transcript = get_youtube_transcript(youtube_url)
            step_times['youtube_transcript'] = time.time() - t0
            if not transcript:
                return jsonify({"error": "Failed to get YouTube transcript"}), 400

        if not transcript:
            return jsonify({"error": "Transcription failed"}), 500

        t0 = time.time()
        summary = summarize_text(transcript)
        step_times['summarize'] = time.time() - t0
        if not summary:
            return jsonify({"error": "Summarization failed"}), 500

        t0 = time.time()
        questions = generate_questions_from_summary(summary)
        step_times['question_gen'] = time.time() - t0

        # Save files
        transcript_file = save_to_file(transcript, "transcript.txt")
        summary_file = save_to_file(summary, "summary.txt")
        questions_file = save_to_file("\n".join(questions), "questions.txt")

        total_time = time.time() - start_time
        logger.info("--- Processing Timings (seconds) ---")
        for k, v in step_times.items():
            logger.info("%s: %.2fs", k, v)
        logger.info("Total: %.2fs", total_time)
        logger.info("-----------------------------------")

        # Return file paths and the actual content
        return jsonify({
            "transcript_content": transcript,
            "summary_content": summary,
            "qa_content": questions,
            "transcript_file": f"/download/{os.path.basename(transcript_file)}",
            "summary_file": f"/download/{os.path.basename(summary_file)}",
            "qa_file": f"/download/{os.path.basename(questions_file)}"
        })

    except Exception as e:
        logger.error("Error in processing: %s", e)
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    filepath = os.path.join("temp", filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    # For local testing only. Render/Gunicorn will run the app in production.
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=False)
