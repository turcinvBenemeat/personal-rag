# personal-rag — x86 / macOS (CPU; add --gpus all for NVIDIA GPU)
#
# Build from the project root:
#   docker build -f docker/Dockerfile -t personal-rag:latest .
#   make build

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -e . --no-deps

COPY src/ ./src/
COPY tests/ ./tests/
COPY config.yaml .env.example ./

ENV HF_HOME=/data/hf-cache
ENV TRANSFORMERS_CACHE=/data/hf-cache
ENV RAG_INDEX_PATH=/data/chroma

ENTRYPOINT ["python", "-m"]
CMD ["rag.query", "--help"]
