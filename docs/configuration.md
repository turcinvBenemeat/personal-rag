# Configuration Reference

## config.yaml

All settings live in `config.yaml` at the project root. User-specific paths can be overridden via environment variables without editing this file — see [Environment variables](#environment-variables) below.

```yaml
vault_path: '/path/to/Obsidian Vault/'
index_path: "./chroma_db"
log_path: "./logs/rag.log"
collection_name: "obsidian_markdown"

exclude_dirs:
  - ".obsidian"
  - ".trash"
  - "Resources/_catalog"
  - "Attachments"
  - "Archive"
  - "Templates"
  - ".git"

exclude_files:
  - ".DS_Store"
  - "CLAUDE.md"
  - "Dashboard.md"
  - "Home.md"
  - "README.md"

chunk_max_chars: 1200
chunk_overlap_chars: 150
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
embedding_batch_size: 16
markdown_workers: 1
pdf_workers: 1
embedding_workers: 1   # legacy, reserved

pdf_sources:
  - path: '/path/to/Books/'
    type: book
  - path: '/path/to/Resources/'
    type: resource
```

### Field reference

| Key | Default | Description |
|---|---|---|
| `vault_path` | — | Obsidian vault root. Supports `~` and spaces. |
| `index_path` | `./chroma_db` | ChromaDB persistent storage directory. |
| `log_path` | `./logs/rag.log` | App log file (rotating, 5 MB × 3). Console output is unaffected. |
| `log_db_path` | `<log_path>.sqlite` | Structured SQLite log DB (`logs` table). Defaults alongside `log_path`. |
| `collection_name` | `obsidian_markdown` | ChromaDB collection name. |
| `exclude_dirs` | see above | Vault subdirectories to skip during indexing. |
| `exclude_files` | `.DS_Store`, `CLAUDE.md`, `Dashboard.md`, `Home.md`, `README.md` | Filenames to skip regardless of directory (e.g. vault hub/navigation pages). |
| `chunk_max_chars` | `1200` | Maximum characters per chunk. Smaller = lower per-file RAM peak. |
| `chunk_overlap_chars` | `150` | Character overlap between consecutive chunks. |
| `embedding_model` | `all-MiniLM-L6-v2` | SentenceTransformers model name or HuggingFace path. |
| `embedding_batch_size` | `16` | Chunks per `model.encode()` call. Keep at 16 on Jetson (8 GB unified RAM). |
| `markdown_workers` | `1` | ThreadPoolExecutor threads for MD extraction. 1 = sequential. |
| `pdf_workers` | `1` | ThreadPoolExecutor threads for PDF extraction. Keep at 1 on Jetson. |
| `pdf_sources` | `[]` | List of `{path, type}` PDF source directories. `type` is stored as chunk metadata. |

## Environment variables

Override any path without editing `config.yaml`. Copy `.env.example` to `.env` — it is loaded automatically at startup via `python-dotenv`.

| Variable | Overrides | Notes |
|---|---|---|
| `RAG_VAULT_PATH` | `vault_path` | |
| `RAG_PDF_BOOKS_PATH` | pdf_sources entry with `type: book` | |
| `RAG_PDF_RESOURCES_PATH` | pdf_sources entry with `type: resource` | |
| `RAG_INDEX_PATH` | `index_path` | Set to `/data/chroma` automatically in Docker. |
| `RAG_LOG_PATH` | `log_path` | Set to `/data/logs/rag.log` automatically in Docker (bind-mounted to `./logs`). |
| `RAG_LOG_DB_PATH` | `log_db_path` | SQLite log DB path. Defaults to the text log path with a `.sqlite` suffix. |
| `RAG_CONFIG_PATH` | Path to `config.yaml` itself | Useful when running from a directory other than the project root. |

### .env example

```bash
RAG_VAULT_PATH=/Volumes/Drive/mindmap/Career Knowledge Base/
RAG_PDF_BOOKS_PATH=/Volumes/Drive/mindmap/Books/
RAG_PDF_RESOURCES_PATH=/Volumes/Drive/mindmap/Resources/
```

## Tuning for different hardware

| Hardware | `chunk_max_chars` | `embedding_batch_size` | `markdown_workers` | `pdf_workers` |
|---|---|---|---|---|
| Jetson Orin Nano Super (8 GB) | 1200 | 16 | 1 | 1 |
| Desktop (16+ GB RAM, no GPU) | 1800 | 32 | 4 | 2 |
| Desktop (NVIDIA GPU, 8+ GB VRAM) | 1800 | 64 | 4 | 2 |
