# personal-rag

Local semantic search over an Obsidian knowledge vault and PDF book library. Indexes Markdown notes and PDFs into a local ChromaDB vector store and retrieves relevant chunks by natural language query.

## What it indexes

- **Obsidian vault** — Markdown notes with YAML frontmatter (type, domain, status, tags, wikilinks)
- **PDF books** — technical books from `mindmap/Books/`
- **PDF resources** — papers, guides, and reference materials from `mindmap/Resources/`

All content lives in Google Drive and is indexed locally. Nothing is sent to any cloud service.

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) — for environment management

## Setup

```bash
# Clone / enter the project
cd personal-rag

# Create virtual environment and install dependencies
uv venv .venv
uv pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` before first run:

```yaml
vault_path: '/path/to/your/Obsidian Vault/'   # Obsidian vault root

pdf_sources:
  - path: '/path/to/Books/'
    type: book
  - path: '/path/to/Resources/'
    type: resource
```

All other settings (chunk size, embedding model, excluded dirs) have sensible defaults.

## Usage

### Index your content

```bash
.venv/bin/python index_obsidian.py
```

Indexing is parallelised: 16 threads for text extraction, then GPU encoding if CUDA is available, or multi-process CPU embedding otherwise. Expect ~5–10 minutes for a large vault + book library on first run.

### Query

```bash
.venv/bin/python query_obsidian.py "What do I know about Kubernetes?"
.venv/bin/python query_obsidian.py "How should I handle secrets in Python?" 12
```

The optional trailing number controls how many results to return (default: 8).

**Output per result:**
- Title and section heading
- File path (note or PDF filename)
- Type, domain, status metadata
- Semantic distance score (lower = more relevant)
- Chunk text preview

### Smoke test retrieval quality

```bash
.venv/bin/python test_queries.py           # run all 13 test queries
.venv/bin/python test_queries.py devops    # filter by keyword
```

Run this after every reindex to catch quality regressions. Results below distance 0.75 are marked OK; above are marked WEAK.

## How it works

```
Obsidian vault (.md)  ─┐
                        ├─► Text extraction (16 threads)
PDF books & resources  ─┘         │
                                   ▼
                        Chunk by heading + character count
                        (1800 chars, 250 overlap)
                                   │
                                   ▼
                        Embed with all-MiniLM-L6-v2
                        (CUDA single-process or CPU ProcessPoolExecutor)
                                   │
                                   ▼
                        ChromaDB (local, ./chroma_db)
                                   │
                          query_obsidian.py
                                   │
                                   ▼
                        Top-N chunks by cosine similarity
```

## Metadata per chunk

| Field | Source | Example |
|---|---|---|
| `path` | file path | `Knowledge/DevOps/K3s.md` |
| `title` | frontmatter or PDF metadata | `K3s Adoption Decision Framework` |
| `heading` | Markdown heading or page range | `Summary` / `p.12-15` |
| `type` | frontmatter or pdf_source type | `Knowledge`, `book`, `resource` |
| `domain` | frontmatter | `DevOps` |
| `status` | frontmatter | `processed` |
| `source` | frontmatter or `pdf` | `ChatGPT`, `pdf` |
| `tags` | frontmatter | `kubernetes, containers` |
| `wikilinks` | extracted from body | `K3s, Docker` |

## Project files

| File | Purpose |
|---|---|
| `index_obsidian.py` | Parallel indexer — MD + PDF → ChromaDB |
| `query_obsidian.py` | Semantic query CLI |
| `test_queries.py` | Retrieval smoke tests |
| `config.yaml` | Vault paths, PDF sources, chunk settings |
| `requirements.txt` | Pinned Python dependencies |

## Known issues

**ChromaDB telemetry warnings** — ChromaDB 0.6.3 and posthog 7.x have a signature mismatch. Both scripts suppress it with a one-line monkey-patch. Safe to ignore; do not remove the patch when upgrading chromadb until confirmed fixed.

**Encrypted PDFs** — a small number of PDFs require AES decryption (`cryptography` package). These are skipped automatically with a warning.

**Unresolved Obsidian template vars** — notes with `{{DATE}}` or similar unfilled template placeholders in their frontmatter are handled by stripping `{{...}}` before YAML parsing.

## Next phase

`answer_obsidian.py` — retrieves top-N chunks from ChromaDB and calls the Claude API to produce a grounded answer with citations to source notes and books. No agent framework; one script, ~60 lines.
