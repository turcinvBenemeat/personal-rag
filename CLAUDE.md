# personal-rag

Local Markdown-only retrieval index for an Obsidian knowledge vault stored in Google Drive.

## What this project does

Embeds Obsidian notes as vector chunks in a local ChromaDB, and retrieves relevant chunks by semantic query. It is a retrieval layer — not a full RAG pipeline yet. Answer generation (calling Claude API) is the planned next phase.

## Environment

**Always use the local `.venv`, never the system Python or Conda.**

```bash
# Run indexer
.venv/bin/python index_obsidian.py

# Run a query
.venv/bin/python query_obsidian.py "your question"
.venv/bin/python query_obsidian.py "your question" 12    # optional: number of results (default 8)

# Recreate environment from scratch
uv venv .venv
uv pip install -r requirements.txt
```

Do not use bare `python` or `python3` — the Conda base environment will be picked up instead of `.venv`.

## Project layout

```
personal-rag/
├── .venv/               # local virtualenv — never commit
├── chroma_db/           # ChromaDB data — never commit
├── config.yaml          # vault path, model, chunk settings
├── index_obsidian.py    # indexes Obsidian vault into ChromaDB
├── query_obsidian.py    # semantic query CLI
├── requirements.txt     # pinned dependencies
└── CLAUDE.md            # this file
```

## config.yaml

- `vault_path`: absolute path to the Obsidian vault (Google Drive, has spaces — handled by `Path.expanduser().resolve()`)
- `index_path`: ChromaDB storage directory (`./chroma_db`)
- `collection_name`: `obsidian_markdown`
- `exclude_dirs`: `.obsidian`, `.trash`, `Resources/_catalog`, `Attachments`, `Archive`, `.git`
- `exclude_files`: `.DS_Store`, `CLAUDE.md`
- `chunk_max_chars`: 1800
- `chunk_overlap_chars`: 250
- `embedding_model`: `sentence-transformers/all-MiniLM-L6-v2`

## ChromaDB state

- Collection: `obsidian_markdown`
- ~6,269 chunks (as of 2026-06-04)
- Metadata fields per chunk: `path`, `title`, `heading`, `type`, `domain`, `status`, `source`, `tags`, `wikilinks`
- Reindexing deletes and recreates the collection — this is intentional for MVP

## Known issue: telemetry noise

ChromaDB 0.6.3 + posthog 7.x have a signature mismatch. Both scripts suppress it via:
```python
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
import posthog as _posthog
_posthog.capture = lambda *a, **kw: None
```
This should be kept. Do not remove it when upgrading chromadb until confirmed fixed.

## Key dependencies (pinned)

| Package | Version |
|---|---|
| chromadb | 0.6.3 |
| sentence-transformers | 3.3.1 |
| python-frontmatter | 1.3.0 |
| PyYAML | 6.0.3 |

## Indexing logic (index_obsidian.py)

1. Reads all `.md` files from vault via `rglob`, skips excluded dirs/files
2. Parses YAML frontmatter with `python-frontmatter`; skips notes with empty body
3. Splits each note by heading into sections
4. Chunks each section by character count with overlap
5. Computes SHA-256 stable IDs from `(rel_path, section_index, chunk_index, chunk[:80])`
6. Embeds in batches of 128 with `all-MiniLM-L6-v2`
7. Upserts into ChromaDB (delete-then-recreate collection)

## Query logic (query_obsidian.py)

1. Embeds the query string with the same model
2. Queries ChromaDB for top-N chunks (default 8, accepts trailing int arg)
3. Prints title, heading, path, type, domain, status, distance, and chunk text (first 1200 chars for display)
4. Full chunk text is in `results["documents"][0]` — use that for LLM context, not the truncated display

## Next phase

Build `answer_obsidian.py`:
- Retrieves top-N chunks via ChromaDB
- Calls Claude API (`claude-sonnet-4-6`) with chunks as context
- Returns grounded answer with cited note paths

No agent framework needed. One script, ~60 lines, `anthropic` SDK only.
