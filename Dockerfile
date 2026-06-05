# personal-rag — x86 / macOS (CPU; add --gpus all for NVIDIA GPU)
#
# Build:  docker build -t personal-rag:latest .
# Or:     make build
#
# Volumes expected at runtime (via docker-compose or -v flags):
#   /data/chroma   — ChromaDB persistent storage
#   /data/hf-cache — HuggingFace model cache
#   /vault         — Obsidian vault (read-only)
#   /books         — PDF books dir (read-only)
#   /resources     — PDF resources dir (read-only)

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml utils.py index_obsidian.py query_obsidian.py test_queries.py .env.example ./

# HuggingFace + ChromaDB data dirs (override via env vars or volume mounts)
ENV HF_HOME=/data/hf-cache
ENV TRANSFORMERS_CACHE=/data/hf-cache
ENV RAG_INDEX_PATH=/data/chroma

ENTRYPOINT ["python"]
CMD ["query_obsidian.py", "--help"]
