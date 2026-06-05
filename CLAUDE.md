# personal-rag

Local retrieval index for an Obsidian knowledge vault + PDF books and resources stored in Google Drive. Embeds content as vector chunks in ChromaDB and retrieves by semantic query.

## What this project does

- Indexes Obsidian Markdown notes and PDF books/resources into a local ChromaDB collection
- Retrieves relevant chunks by semantic similarity query
- Fully offline — no cloud APIs used for indexing or retrieval

## Environment

**Always use the local `.venv`, never the system Python or Conda.**

```bash
# First-time setup
make install          # creates .venv and installs package + deps

# Run indexer
.venv/bin/rag-index

# Run a query
.venv/bin/rag-query "your question"
.venv/bin/rag-query "your question" -n 12    # optional: number of results (default 8)
.venv/bin/rag-query "Kubernetes" --domain DevOps

# Run retrieval smoke tests
.venv/bin/python tests/test_queries.py
.venv/bin/python tests/test_queries.py kubernetes   # filter by keyword

# Recreate environment from scratch
uv venv .venv
uv pip install -r requirements.txt
uv pip install -e . --no-deps
```

Do not use bare `python` or `python3` — the Conda base environment will be picked up instead of `.venv`.

## Project layout

```
personal-rag/
├── src/
│   └── rag/
│       ├── __init__.py
│       ├── utils.py          # load_config, telemetry suppression, .env loading
│       ├── indexer.py        # streaming indexer (MD + PDF → ChromaDB); entry: rag-index
│       └── query.py          # semantic query CLI; entry: rag-query
├── tests/
│   └── test_queries.py       # 13-query smoke tests across all vault domains
├── Dockerfile                # x86 / macOS container image
├── Dockerfile.jetson         # Jetson JetPack 6.2 container image (build on Jetson)
├── docker-compose.yml        # x86 Compose with volume mounts
├── docker-compose.jetson.yml # Jetson Compose (runtime: nvidia)
├── .venv/                    # local virtualenv — never commit
├── chroma_db/                # ChromaDB data — never commit
├── .env                      # path overrides — never commit (see .env.example)
├── .env.example              # template for .env
├── .dockerignore
├── config.yaml               # vault path, pdf sources, model, chunk settings
├── pyproject.toml            # package definition and CLI entry points
├── Makefile                  # shortcuts: make install/index/query/test, make build/docker-*/jetson-*
├── requirements-direct.txt   # direct deps only — use to regenerate lockfile
├── requirements.txt          # full pinned lockfile — x86 / macOS
├── requirements-jetson.txt   # pinned deps — Jetson JetPack 6.2 (aarch64, CUDA 12.6)
├── docs/
│   ├── architecture.md       # pipeline walkthrough, chunk IDs, telemetry workaround
│   ├── configuration.md      # full config.yaml reference and env var overrides
│   └── jetson.md             # Jetson install, Docker, memory budget, GPU constraints
├── README.md                 # setup and usage guide
└── CLAUDE.md                 # this file
```

## Key dependencies (pinned)

| Package | Version |
|---|---|
| chromadb | 0.6.3 |
| sentence-transformers | 3.3.1 |
| python-frontmatter | 1.3.0 |
| PyYAML | 6.0.3 |
| pypdf | 6.12.2 |

## ChromaDB state

- Collection: `obsidian_markdown`
- ~7,600 MD chunks + PDF chunks from 175 books and 35 resources (as of 2026-06-04)
- Metadata fields: `path`, `title`, `heading`, `type`, `domain`, `status`, `source`, `tags`, `wikilinks`
- `source: pdf` distinguishes PDF chunks from Markdown chunks
- Reindexing deletes and recreates the collection — intentional for MVP

## Deeper reference

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Pipeline walkthrough, chunk IDs, ChromaDB state, telemetry workaround |
| [docs/configuration.md](docs/configuration.md) | All `config.yaml` fields, env var overrides, hardware tuning table |
| [docs/jetson.md](docs/jetson.md) | Jetson-specific install, Docker, memory budget, IPC constraints |
