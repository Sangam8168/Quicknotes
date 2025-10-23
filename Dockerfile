# Use a Python base image with a Python version compatible with CPU PyTorch wheels
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PORT=10000
WORKDIR /app

# Install system dependencies (ffmpeg, libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install CPU-only torch wheel compatible with Python 3.10
# If you want a different torch version, pick the matching wheel for your Python version.
RUN pip install "torch==2.2.0+cpu" --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining dependencies (requirements.txt should NOT include torch)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Ensure we have the latest youtube-transcript-api (explicitly)
RUN pip install --upgrade youtube-transcript-api

# Copy application files
COPY . /app

# Pre-download NLTK punkt to avoid runtime downloads
RUN python -m nltk.downloader punkt

EXPOSE ${PORT}

# Start with gunicorn binding to the $PORT that Render provides
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:${PORT}", "--workers", "1", "--threads", "4"]
