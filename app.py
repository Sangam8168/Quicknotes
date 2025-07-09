from flask import Flask, render_template, request, jsonify, send_file
import whisper
from transformers import pipeline
import os
import uuid
from nltk.tokenize import sent_tokenize
import nltk
import warnings
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import PyPDF2  # Added for PDF support

# Suppress Whisper warnings
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")

# Ensure NLTK resources are downloaded
nltk.download('punkt')
app = Flask(__name__, static_folder='static')

# Add FFmpeg executable path
os.environ[
    "PATH"] = r"C:\Users\Ritika\Downloads\QuickNotes\QuickNotes\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin;" + \
              os.environ["PATH"]

# Load models
print("Loading models...")
transcribe_model = whisper.load_model("base")  # Better Whisper model for transcription
summarizer = pipeline("summarization", model="facebook/bart-large-cnn",
                      tokenizer="facebook/bart-large-cnn")  # Advanced summarizer
qa_generator = pipeline("text2text-generation", model="google/flan-t5-base")  # Enhanced question generation
print("Models loaded.")


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
        print(f"Error extracting video ID: {e}")
        return None


def get_youtube_transcript(url):
    """Get transcript from YouTube video using youtube_transcript_api."""
    try:
        video_id = extract_video_id(url)
        if not video_id:
            raise ValueError("Could not extract video ID from URL")

        print(f"Getting transcript for video ID: {video_id}")
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

        # Combine all text parts into a single transcript
        full_transcript = " ".join([item['text'] for item in transcript_list])
        return full_transcript
    except Exception as e:
        print(f"YouTube transcript error: {e}")
        return None


def download_youtube_audio(url):
    """Download audio from YouTube videos - kept for audio processing if needed."""
    try:
        video_id = extract_video_id(url)
        if not video_id:
            raise ValueError("Unable to extract YouTube video ID.")

        # Generate a unique filename
        filename = f"{uuid.uuid4()}.mp4"
        output_path = "temp"

        # Create the directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # For now, just return the ID as we'll be using transcript API instead
        return video_id
    except Exception as e:
        print(f"YouTube download error: {e}")
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
        print(f"PDF extraction error: {e}")
        return None


def transcribe_audio(filepath):
    print("Transcribing...")
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        print("Invalid or empty audio file.")
        return None
    try:
        result = transcribe_model.transcribe(filepath)
        print("Transcription Result:", result["text"])
        return result["text"]
    except Exception as e:
        print("Error during transcription:", e)
        return None


def preprocess_text(text):
    print("Preprocessing transcript...")
    try:
        sentences = sent_tokenize(text)
        cleaned_sentences = [sentence.strip() for sentence in sentences if len(sentence.split()) > 3]
        preprocessed_text = " ".join(cleaned_sentences)
        print("Preprocessed Text:", preprocessed_text)
        return preprocessed_text
    except Exception as e:
        print("Error during preprocessing:", e)
        return text


def summarize_text(text):
    print("Summarizing...")
    try:
        preprocessed_text = preprocess_text(text)
        chunks = [preprocessed_text[i:i + 1000] for i in range(0, len(preprocessed_text), 1000)]
        summaries = []
        for chunk in chunks:
            input_length = len(chunk.split())  # Count words
            max_len = max(50, int(input_length * 0.8))  # Adjust max_length to 80% of input length
            summary = summarizer(chunk, max_length=max_len, min_length=30, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        summarized_text = " ".join(summaries)
        return summarized_text
    except Exception as e:
        print("Error during summarization:", e)
        return "Error during summarization."


import traceback

def generate_questions_from_summary(summary):
    print("Generating questions from summary...")
    print("Summary input:", summary)
    try:
        sentences = sent_tokenize(summary)
        print("Tokenized sentences:", sentences)
        if not sentences:
            print("No sentences found in summary.")
            return ["No questions generated: summary was empty."]
        questions = []
        for sentence in sentences[:5]:
            qa = qa_generator(
                f"Generate a question based on: {sentence}",
                max_length=64,
                num_return_sequences=1,
                do_sample=True
            )[0]['generated_text']
            questions.append(qa)
        return questions
    except Exception as e:
        print("Error generating questions:", e)
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
        return jsonify({"error": str(e)}), 500


@app.route('/process', methods=['POST'])
def process():
    import time
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
            # Check if PDF
            if filename.lower().endswith('.pdf'):
                t0 = time.time()
                transcript = extract_text_from_pdf(filepath)
                step_times['pdf_extract'] = time.time() - t0
                if not transcript:
                    return jsonify({"error": "Failed to extract text from PDF"}), 400
            else:
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
        print("--- Processing Timings (seconds) ---")
        for k, v in step_times.items():
            print(f"{k}: {v:.2f}s")
        print(f"Total: {total_time:.2f}s")
        print("-----------------------------------")

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
        print("Error:", e)
        return jsonify({"error": str(e)}), 500


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    filepath = os.path.join("temp", filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({"error": "File not found"}), 404


if __name__ == '__main__':
    app.run(debug=True, threaded=False)  # Single-threaded mode for stability