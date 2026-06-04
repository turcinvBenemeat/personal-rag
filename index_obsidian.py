import hashlib
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import posthog as _posthog
_posthog.capture = lambda *a, **kw: None  # chromadb 0.6.x / posthog 7.x signature mismatch

import torch

import chromadb
import frontmatter
import yaml
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_IO_WORKERS = 16           # threads for parallel text extraction (I/O bound)
_CPU_CORES = multiprocessing.cpu_count()  # processes for embedding (CPU bound)


_worker_model = None  # one model instance per worker process


def _init_worker(model_name: str) -> None:
    """
    Called once when a worker process starts.
    Pins the process to 1 OpenMP thread and loads the model into a global
    so it is reused across all batches — no repeated cold-start overhead.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    from sentence_transformers import SentenceTransformer
    global _worker_model
    _worker_model = SentenceTransformer(model_name)


def _embed_batch(docs: list) -> np.ndarray:
    """Embed one batch using the already-loaded worker model."""
    return _worker_model.encode(docs, normalize_embeddings=True, batch_size=64)


def log(msg: str) -> None:
    """Print with immediate flush so progress is visible even when piped."""
    print(msg, flush=True)


def load_config():
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def should_exclude(path: Path, vault_path: Path, config: dict) -> bool:
    rel = path.relative_to(vault_path).as_posix()

    for excluded in config.get("exclude_dirs", []):
        excluded = excluded.strip("/")
        if rel == excluded or rel.startswith(excluded + "/"):
            return True

    if path.name in config.get("exclude_files", []):
        return True

    return False


def extract_wikilinks(text: str):
    return sorted(set(re.findall(r"\[\[([^\]|#]+)", text)))


def split_by_headings(text: str):
    sections = []
    current_heading = "Document"
    current_lines = []

    for line in text.splitlines():
        if line.startswith("#"):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
                current_lines = []
            current_heading = line.strip("#").strip() or "Document"
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    return [(h, t) for h, t in sections if t.strip()]


def chunk_text(text: str, max_chars: int, overlap: int):
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks


def stable_id(*parts):
    raw = "::".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def clean_pdf_title(filename: str) -> str:
    """Fallback title from filename when PDF metadata is unavailable."""
    name = Path(filename).stem
    name = re.sub(r"[_\-]?(v\d+|[23456]e|[23456]rdedition|[23456]thedition)$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[_\-]+", " ", name)
    return name.strip().title()


# ---------------------------------------------------------------------------
# Per-file extraction functions (run in threads)
# ---------------------------------------------------------------------------

def extract_md_file(md_file: Path, vault_path: Path, config: dict, max_chars: int, overlap: int):
    """Parse one Markdown file and return (ids, documents, metadatas) or ([], [], [])."""
    rel_path = md_file.relative_to(vault_path).as_posix()

    try:
        raw = md_file.read_text(encoding="utf-8", errors="ignore")
        raw = re.sub(r"\{\{[^}]+\}\}", "", raw)  # strip unresolved Obsidian template vars
        parsed = frontmatter.loads(raw)
        body = parsed.content.strip()
        meta = dict(parsed.metadata)
    except Exception as exc:
        return [], [], [], f"skip parse error: {rel_path}: {exc}"

    if not body:
        return [], [], [], None

    title = str(meta.get("title") or md_file.stem)
    note_type = str(meta.get("type") or "")
    domain = str(meta.get("domain") or "")
    status = str(meta.get("status") or "")
    source = str(meta.get("source") or "")

    tags_value = meta.get("tags") or []
    tags = ", ".join(str(t) for t in tags_value) if isinstance(tags_value, list) else str(tags_value)

    wikilinks = ", ".join(extract_wikilinks(body))
    sections = split_by_headings(body)

    ids, documents, metadatas = [], [], []

    for section_index, (heading, section_text) in enumerate(sections):
        for chunk_index, chunk in enumerate(chunk_text(section_text, max_chars=max_chars, overlap=overlap)):
            ids.append(stable_id(rel_path, section_index, chunk_index, chunk[:80]))
            documents.append(chunk)
            metadatas.append({
                "path": rel_path,
                "title": title,
                "heading": heading,
                "type": note_type,
                "domain": domain,
                "status": status,
                "source": source,
                "tags": tags,
                "wikilinks": wikilinks,
            })

    return ids, documents, metadatas, None


def extract_pdf_file(pdf_path: Path, source_type: str, max_chars: int, overlap: int):
    """Extract text from one PDF and return (ids, documents, metadatas) or ([], [], [])."""
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        return [], [], [], f"skip pdf read error: {pdf_path.name}: {exc}"

    pdf_meta = reader.metadata
    title = (pdf_meta.title.strip() if pdf_meta and pdf_meta.title else None) or clean_pdf_title(pdf_path.name)

    ids, documents, metadatas = [], [], []
    block_text = ""
    block_start = 1

    def flush_block(block_text, block_start, end_page):
        for chunk_index, chunk in enumerate(chunk_text(block_text.strip(), max_chars, overlap)):
            heading = f"p.{block_start}" if block_start == end_page else f"p.{block_start}-{end_page}"
            ids.append(stable_id(str(pdf_path), block_start, chunk_index, chunk[:80]))
            documents.append(chunk)
            metadatas.append({
                "path": pdf_path.name,
                "title": title,
                "heading": heading,
                "type": source_type,
                "domain": "",
                "status": "",
                "source": "pdf",
                "tags": "",
                "wikilinks": "",
            })

    for page_num, page in enumerate(reader.pages, 1):
        try:
            page_text = (page.extract_text() or "").strip()
            page_text = page_text.encode("utf-8", errors="replace").decode("utf-8")
        except Exception:
            continue

        if len(page_text) < 40:
            continue

        block_text += "\n\n" + page_text

        if len(block_text) >= max_chars:
            flush_block(block_text, block_start, page_num)
            block_text = ""
            block_start = page_num + 1

    if block_text.strip():
        flush_block(block_text, block_start, len(reader.pages))

    return ids, documents, metadatas, None


# ---------------------------------------------------------------------------
# Parallel collection helpers
# ---------------------------------------------------------------------------

def collect_parallel(futures_map, label: str):
    """
    Drain a dict of {future: name} as futures complete.
    Returns (all_ids, all_docs, all_metas).
    Prints per-file progress, errors, and a summary line.
    """
    all_ids, all_docs, all_metas = [], [], []
    errors = 0
    total = len(futures_map)
    done = 0

    for future in as_completed(futures_map):
        name = futures_map[future]
        ids, docs, metas, err = future.result()
        done += 1
        if err:
            log(f"  [{done}/{total}] SKIP {name}: {err.split(':', 1)[-1].strip()}")
            errors += 1
        else:
            log(f"  [{done}/{total}] {name}  ({len(docs)} chunks)")
        all_ids.extend(ids)
        all_docs.extend(docs)
        all_metas.extend(metas)

    log(f"{label}: {len(all_docs)} chunks total ({errors} skipped)")
    return all_ids, all_docs, all_metas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = load_config()

    vault_path = Path(config["vault_path"]).expanduser().resolve()
    index_path = config.get("index_path", "./chroma_db")
    collection_name = config.get("collection_name", "obsidian_markdown")
    max_chars = int(config.get("chunk_max_chars", 1800))
    overlap = int(config.get("chunk_overlap_chars", 250))
    model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

    if not vault_path.exists():
        raise RuntimeError(f"Vault path does not exist: {vault_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    log(f"Vault: {vault_path}")
    log(f"Embedding model: {model_name}")
    log(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else f" ({_CPU_CORES} cores, {_IO_WORKERS} I/O threads)"))

    client = chromadb.PersistentClient(
        path=index_path,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )

    try:
        client.delete_collection(collection_name)
        log(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass

    collection = client.create_collection(name=collection_name)

    all_ids, all_docs, all_metas = [], [], []

    # --- Markdown files (parallel) ---
    md_files = [f for f in sorted(vault_path.rglob("*.md")) if not should_exclude(f, vault_path, config)]
    log(f"Markdown files to index: {len(md_files)}")

    with ThreadPoolExecutor(max_workers=_IO_WORKERS) as pool:
        futures = {
            pool.submit(extract_md_file, f, vault_path, config, max_chars, overlap): f.name
            for f in md_files
        }
        ids, docs, metas = collect_parallel(futures, "Markdown")
        all_ids.extend(ids)
        all_docs.extend(docs)
        all_metas.extend(metas)

    # --- PDF sources (parallel per source) ---
    for pdf_source in config.get("pdf_sources", []):
        pdf_dir = Path(pdf_source["path"]).expanduser().resolve()
        source_type = pdf_source.get("type", "resource")

        if not pdf_dir.exists():
            log(f"Warning: pdf_source path does not exist, skipping: {pdf_dir}")
            continue

        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        log(f"PDF source [{source_type}]: {len(pdf_files)} files — {pdf_dir.name}")

        with ThreadPoolExecutor(max_workers=_IO_WORKERS) as pool:
            futures = {
                pool.submit(extract_pdf_file, f, source_type, max_chars, overlap): f.name
                for f in pdf_files
            }
            ids, docs, metas = collect_parallel(futures, f"  [{source_type}]")
            all_ids.extend(ids)
            all_docs.extend(docs)
            all_metas.extend(metas)

    log(f"Total chunks (MD + PDF): {len(all_docs)}")

    if not all_docs:
        raise RuntimeError("No documents found to index.")

    # --- Embed and upsert ---
    if device == "cuda":
        # GPU path: single process, large batches — GPU handles parallelism internally.
        # encode_multi_process() is intentionally avoided: Jetson does not support
        # CUDA IPC (uses NvSCI instead) so cross-process tensor sharing fails.
        log(f"Embedding {len(all_docs)} chunks on GPU (single process, batch=512)...")
        model = SentenceTransformer(model_name, device="cuda")
        all_embeddings = model.encode(
            all_docs,
            normalize_embeddings=True,
            batch_size=512,
            show_progress_bar=True,
        )
    else:
        # CPU path: ProcessPoolExecutor — each worker loads the model once
        # (initializer) and is fed small batches, with OMP_NUM_THREADS=1
        # so N workers each own exactly 1 core.
        embed_batch_size = 256
        batches = [all_docs[i:i + embed_batch_size] for i in range(0, len(all_docs), embed_batch_size)]
        log(f"Embedding {len(all_docs)} chunks across {_CPU_CORES} CPU workers ({len(batches)} batches)...")
        with ProcessPoolExecutor(
            max_workers=_CPU_CORES,
            initializer=_init_worker,
            initargs=(model_name,),
        ) as executor:
            embedding_chunks = list(executor.map(_embed_batch, batches))
        all_embeddings = np.vstack(embedding_chunks)

    log("Embedding done. Upserting to ChromaDB...")

    upsert_batch = 512  # ChromaDB handles larger batches fine for upsert
    for start in range(0, len(all_docs), upsert_batch):
        end = start + upsert_batch
        collection.add(
            ids=all_ids[start:end],
            documents=all_docs[start:end],
            embeddings=all_embeddings[start:end].tolist(),
            metadatas=all_metas[start:end],
        )
        log(f"Upserted {min(end, len(all_docs))}/{len(all_docs)} chunks")

    log("Indexing complete.")


if __name__ == "__main__":
    main()
