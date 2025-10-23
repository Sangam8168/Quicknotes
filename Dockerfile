# Dockerfile to deploy on Render with ffmpeg and CPU PyTorch
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
# Default container port; Render will set $PORT at runtime, we set a sane default
ENV PORT=10000
WORKDIR /app

# Install system dependencies (ffmpeg, libsndfile for audio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    libsndfile1 \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install CPU-only torch wheel (adjust version as needed)
# This uses the PyTorch CPU wheel index. If you want GPU support, adjust accordingly.
RUN pip install "torch==2.2.0+cpu" --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install rest (requirements.txt excludes torch)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app
COPY . /app

# Pre-download NLTK punkt to avoid runtime downloads
RUN python -m nltk.downloader punkt

# Expose the default port (Render maps incoming traffic)
EXPOSE ${PORT}

# Use gunicorn and bind to $PORT (shell form allows env var expansion)
CMD gunicorn app:app -b 0.0.0.0:$PORT --workers 1 --threads 4
