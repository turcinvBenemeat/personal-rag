# personal-rag

Local retrieval index for an Obsidian knowledge vault + PDF books and resources stored in Google Drive. Embeds content as vector chunks in ChromaDB and retrieves by semantic query.

## What this project does

- Indexes Obsidian Markdown notes and PDF books/resources into a local ChromaDB collection
- Retrieves relevant chunks by semantic similarity query
- Retrieval layer only — answer generation (Claude API) is the planned next phase

## Environment

**Always use the local `.venv`, never the system Python or Conda.**

```bash
# Run indexer
.venv/bin/python index_obsidian.py

# Run a query
.venv/bin/python query_obsidian.py "your question"
.venv/bin/python query_obsidian.py "your question" 12    # optional: number of results (default 8)

# Run retrieval smoke tests
.venv/bin/python test_queries.py
.venv/bin/python test_queries.py kubernetes             # filter by keyword

# Recreate environment from scratch
uv venv .venv
uv pip install -r requirements.txt
```

Do not use bare `python` or `python3` — the Conda base environment will be picked up instead of `.venv`.

## Project layout

```
personal-rag/
├── .venv/                    # local virtualenv — never commit
├── chroma_db/                # ChromaDB data — never commit
├── config.yaml               # vault path, pdf sources, model, chunk settings
├── index_obsidian.py         # parallel indexer (MD + PDF → ChromaDB)
├── query_obsidian.py         # semantic query CLI
├── test_queries.py           # retrieval smoke tests across all domains
├── requirements.txt          # pinned deps — x86 / macOS
├── requirements-jetson.txt   # pinned deps — Jetson JetPack 6.2 (aarch64, CUDA 12.6)
├── README.md                 # setup and usage guide
└── CLAUDE.md                 # this file
```

## config.yaml

- `vault_path`: Obsidian vault path (Google Drive with spaces — handled by `Path.expanduser().resolve()`)
- `index_path`: ChromaDB storage directory (`./chroma_db`)
- `collection_name`: `obsidian_markdown`
- `exclude_dirs`: `.obsidian`, `.trash`, `Resources/_catalog`, `Attachments`, `Archive`, `.git`
- `exclude_files`: `.DS_Store`, `CLAUDE.md`
- `chunk_max_chars`: `1200` — smaller chunks reduce per-file peak RAM; increase on high-RAM machines
- `chunk_overlap_chars`: `150`
- `embedding_model`: `sentence-transformers/all-MiniLM-L6-v2`
- `embedding_batch_size`: `16` — chunks per `model.encode()` call; keep small on Jetson (8 GB unified RAM)
- `markdown_workers`: `1` — threads for parallel MD file extraction (I/O bound); 1 = sequential
- `pdf_workers`: `1` — threads for parallel PDF extraction; keep at 1 on Jetson
- `embedding_workers`: `1` — CPU ProcessPoolExecutor workers (legacy, not used by streaming indexer; reserved)
- `pdf_sources`: list of `{path, type}` entries for PDF directories (`book`, `resource`)

## ChromaDB state

- Collection: `obsidian_markdown`
- ~7,600 MD chunks + PDF chunks from 175 books and 35 resources (as of 2026-06-04)
- Metadata fields per chunk: `path`, `title`, `heading`, `type`, `domain`, `status`, `source`, `tags`, `wikilinks`
- `source: pdf` distinguishes PDF chunks from Markdown chunks
- Reindexing deletes and recreates the collection — intentional for MVP

## Known issue: telemetry noise

ChromaDB 0.6.3 + posthog 7.x have a signature mismatch. Both scripts suppress it via:
```python
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
import posthog as _posthog
_posthog.capture = lambda *a, **kw: None
```
Keep this. Do not remove when upgrading chromadb until confirmed fixed.

## Key dependencies (pinned)

| Package | Version |
|---|---|
| chromadb | 0.6.3 |
| sentence-transformers | 3.3.1 |
| python-frontmatter | 1.3.0 |
| PyYAML | 6.0.3 |
| pypdf | 6.12.2 |

## Indexing logic (index_obsidian.py)

**Pipeline is fully streaming — no global accumulation of chunks in RAM.**
Each file is processed end-to-end (extract → embed → upsert → free) before the next file starts.

**Per-file loop:**
1. **Extract** — I/O runs in a `ThreadPoolExecutor` (`markdown_workers` or `pdf_workers` threads):
   - Markdown: reads `.md` via `rglob`, skips excluded dirs/files, strips `{{...}}` Obsidian template vars, splits by heading, chunks by char count with overlap
   - PDF: reads pages into text blocks, chunks by char count; prefers embedded PDF title metadata over filename
2. **Embed** — always runs in the main thread (never multiprocess); `model.encode()` called with `embedding_batch_size` chunks at a time; `torch.cuda.empty_cache()` after each batch on CUDA
3. **Upsert** — each batch is written to ChromaDB immediately after embedding; no full-vault buffer
4. **Free** — `del ids/docs/metas` + `gc.collect()` after each file; `torch.cuda.empty_cache()` after each PDF

**Model loading:** `SentenceTransformer` loaded once at startup, auto-selects `device="cuda"` or `device="cpu"`. `encode_multi_process()` is intentionally avoided — Jetson does not support CUDA IPC (uses NvSCI instead).

**Collection lifecycle:** delete-then-recreate at the start of every full reindex.

**Stable IDs:** SHA-256 of `(path, section_index, chunk_index, chunk[:80])`

## Jetson setup

- **Hardware:** NVIDIA Jetson Orin Nano Super
- **JetPack:** 6.2 — CUDA 12.6, Python 3.10 (not 3.12)
- **PyTorch:** install from Jetson AI Lab index before other deps:
  ```bash
  pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
  pip install -r requirements-jetson.txt
  ```
- **Why not `encode_multi_process`:** Jetson uses NvSCI IPC, not CUDA IPC — cross-process tensor sharing fails. Single-process GPU encoding is the correct path.
- **RAM:** 8 GB unified (CPU + GPU share the same pool). The streaming indexer avoids global chunk accumulation — peak RAM is bounded to one file's chunks at a time. Keep `embedding_batch_size: 16`, `markdown_workers: 1`, `pdf_workers: 1` to stay within budget.
- **ChromaDB:** `0.6.3` publishes `manylinux_2_17_aarch64` wheels — no changes needed.

## Query logic (query_obsidian.py)

1. Embeds the query string with the same model
2. Queries ChromaDB for top-N chunks (default 8, accepts trailing int arg)
3. Prints title, heading, path, type, domain, status, distance, and chunk text (first 1200 chars for display)
4. Full chunk text is in `results["documents"][0]` — use that for LLM context, not the truncated display

## Test queries (test_queries.py)

- 13 pre-defined queries covering all 6 vault domains
- Prints top 3 results per query with distance scores
- Flags WEAK results (distance ≥ 0.75) vs OK
- Run after every reindex to verify retrieval quality
- Add your own queries to `TEST_QUERIES` list as the vault grows

## Next phase

Build `answer_obsidian.py`:
- Retrieves top-N chunks via ChromaDB (MD + PDF)
- Calls Claude API (`claude-sonnet-4-6`) with chunks as context
- Returns grounded answer with cited note paths and book references

No agent framework needed. One script, ~60 lines, `anthropic` SDK only.
