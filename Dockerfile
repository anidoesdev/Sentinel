FROM python:3.11-slim

WORKDIR /app

# libsndfile1: required by soundfile for WAV decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY sentinel/ sentinel/
COPY scripts/ scripts/

RUN pip install --no-cache-dir -e .

# HuggingFace model cache (AST downloads ~500MB on first startup, persisted by volume)
ENV HF_HOME=/cache/huggingface
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "sentinel.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
