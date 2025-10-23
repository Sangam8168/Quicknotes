FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PORT=10000
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install CPU-only PyTorch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Ensure latest youtube-transcript-api
RUN pip install --upgrade youtube-transcript-api

# Copy application files
COPY . /app

# Pre-download NLTK punkt
RUN python -m nltk.downloader punkt

EXPOSE ${PORT}

# Start app with Gunicorn
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:${PORT}", "--workers", "1", "--threads", "4"]
