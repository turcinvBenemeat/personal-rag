# Architecture

## Pipeline overview

```
Obsidian vault (.md)  ─┐
                        ├─► Per-file text extraction (ThreadPoolExecutor)
PDF books & resources  ─┘         │
                                   │  one file at a time (streaming)
                                   ▼
                        Chunk by heading + character count
                        (1200 chars max, 150 overlap)
                                   │
                                   ▼
                        Embed with all-MiniLM-L6-v2
                        (single-process; CUDA on Jetson/GPU, CPU fallback)
                                   │
                                   ▼
                        Upsert batch → ChromaDB (local, ./chroma_db)
                        free memory → next file
                                   │
                          rag-query / python -m rag.query
                                   │
                                   ▼
                        Top-N chunks by cosine similarity
                        optional metadata filter (--domain / --type / --source)
```

## Streaming indexer (`src/rag/indexer.py`)

The pipeline is **fully streaming** — no global accumulation of chunks in RAM. Each file is processed end-to-end (extract → embed → upsert → free) before the next file starts.

### Per-file loop

1. **Extract** — I/O runs in a `ThreadPoolExecutor` (`markdown_workers` or `pdf_workers` threads):
   - Markdown: reads `.md` via `rglob`, skips excluded dirs/files, strips `{{...}}` Obsidian template vars, splits by heading, chunks by char count with overlap
   - PDF: reads pages into text blocks, chunks by char count; prefers embedded PDF title metadata over filename

2. **Embed** — always runs in the main thread (never multiprocess); `model.encode()` called with `embedding_batch_size` chunks at a time; `torch.cuda.empty_cache()` after each batch on CUDA

3. **Upsert** — each batch is written to ChromaDB immediately after embedding; no full-vault buffer

4. **Free** — `del ids/docs/metas` + `gc.collect()` after each file; `torch.cuda.empty_cache()` after each PDF

### Why single-process embedding

`encode_multi_process()` is intentionally avoided — Jetson uses NvSCI IPC (not CUDA IPC), so cross-process tensor sharing fails. Single-process GPU encoding is the correct path for both Jetson and desktop GPUs.

### Stable chunk IDs

SHA-256 of `(path, section_index, chunk_index, chunk)` — the full chunk text is hashed, so identical content always yields the same ID and any edit yields a new one. This is what makes incremental indexing reliable.

### Collection lifecycle (incremental)

The collection is opened with `get_or_create_collection` — never wiped. Each run:

1. Snapshots the IDs already in the collection.
2. For every chunk, records its ID as "seen" and embeds + upserts it only if the ID is not already present (unchanged chunks are skipped — no re-embedding).
3. After all sources are processed, deletes any indexed ID that was not seen this run — pruning chunks from edited files (old content) and from deleted files.

Re-running on an unchanged vault embeds nothing. Changing `chunk_max_chars`/`chunk_overlap_chars` changes every chunk's text and therefore every ID, so the next run re-embeds everything and prunes the old chunks — effectively a clean rebuild.

## Query (`src/rag/query.py`)

1. Embeds the query string with the same model (CPU, no device selection needed for single inference)
2. Builds an optional ChromaDB `where` filter from `--domain`, `--type`, `--source` flags
3. Queries ChromaDB for top-N chunks by cosine similarity
4. Prints results or dumps as JSON (`--json`)

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

## ChromaDB state

- Collection: `obsidian_markdown`
- ~7,600 MD chunks + PDF chunks from 175 books and 35 resources (as of 2026-06-04)
- `source: pdf` distinguishes PDF chunks from Markdown chunks

## Known issue: telemetry noise

ChromaDB 0.6.3 + posthog 7.x have a signature mismatch. `src/rag/utils.py` suppresses it at module import time:

```python
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
import posthog as _posthog
_posthog.capture = lambda *a, **kw: None
```

Do not remove when upgrading chromadb until confirmed fixed.
