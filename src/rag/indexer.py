"""Streaming indexer — Obsidian Markdown + PDF → ChromaDB."""

import gc
import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .utils import load_config, setup_logging  # sets telemetry env var and patches posthog before chromadb loads

import torch
import chromadb
import frontmatter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


logger = logging.getLogger("rag")
log = logger.info  # bound to the shared 'rag' logger; configured by setup_logging() in main()


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
# Per-file extraction (I/O-bound, safe to run in threads)
# ---------------------------------------------------------------------------

def extract_md_file(md_file: Path, vault_path: Path, config: dict, max_chars: int, overlap: int):
    """Parse one Markdown file → (ids, documents, metadatas, error)."""
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
    confidence = str(meta.get("confidence") or "")
    tags_value = meta.get("tags") or []
    tags = ", ".join(str(t) for t in tags_value) if isinstance(tags_value, list) else str(tags_value)
    wikilinks = ", ".join(extract_wikilinks(body))
    sections = split_by_headings(body)

    ids, documents, metadatas = [], [], []
    for section_index, (heading, section_text) in enumerate(sections):
        for chunk_index, chunk in enumerate(chunk_text(section_text, max_chars=max_chars, overlap=overlap)):
            ids.append(stable_id(rel_path, section_index, chunk_index, chunk))
            documents.append(chunk)
            metadatas.append({
                "path": rel_path, "title": title, "heading": heading,
                "type": note_type, "domain": domain, "status": status,
                "source": source, "confidence": confidence,
                "tags": tags, "wikilinks": wikilinks,
            })
    return ids, documents, metadatas, None


def extract_pdf_file(pdf_path: Path, source_type: str, max_chars: int, overlap: int):
    """Extract text from one PDF → (ids, documents, metadatas, error)."""
    try:
        reader = PdfReader(str(pdf_path))
        if reader.is_encrypted:
            reader.decrypt("")
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
            ids.append(stable_id(str(pdf_path), block_start, chunk_index, chunk))
            documents.append(chunk)
            metadatas.append({
                "path": pdf_path.name, "title": title, "heading": heading,
                "type": source_type, "domain": "", "status": "",
                "source": "pdf", "confidence": "",
                "tags": "", "wikilinks": "",
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


def extract_json_doc(json_path: Path, max_chars: int, overlap: int):
    """Index a pre-extracted document JSON (doc-text-extractor `indexed/*.json`).

    Each file carries the full extracted `text` plus enriched metadata
    (title, primary_topic, resource_type, tags, confidence). Reuses the same
    char-window chunker as the other extractors; IDs are content-hashed off the
    stable pre-extracted text, so re-runs are idempotent (unlike live PDF
    parsing). Returns (ids, documents, metadatas, error)."""
    try:
        obj = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], [], [], f"skip json read error: {json_path.name}: {exc}"

    text = (obj.get("text") or "").strip()
    if len(text) < 40:
        return [], [], [], None  # empty / failed extraction — nothing to index

    file_name = str(obj.get("file_name") or json_path.stem)
    tags_value = obj.get("tags") or []
    tags = ", ".join(str(t) for t in tags_value) if isinstance(tags_value, list) else str(tags_value)
    meta_base = {
        "path": file_name,
        "title": str(obj.get("title") or file_name),
        "type": str(obj.get("resource_type") or obj.get("source_group") or "resource"),
        "domain": str(obj.get("primary_topic") or ""),
        "status": "",
        "source": "pdf",  # keep books/resources under the existing `--source pdf` filter
        "confidence": str(obj.get("confidence") or ""),
        "tags": tags,
        "wikilinks": "",
    }

    ids, documents, metadatas = [], [], []
    for chunk_index, chunk in enumerate(chunk_text(text, max_chars, overlap)):
        ids.append(stable_id(file_name, chunk_index, chunk))
        documents.append(chunk)
        metadatas.append({**meta_base, "heading": f"part {chunk_index + 1}"})
    return ids, documents, metadatas, None


# ---------------------------------------------------------------------------
# Streaming embed + upsert
# ---------------------------------------------------------------------------

def embed_and_upsert(model, device, docs, ids, metas, embed_batch_size, collection):
    """Embed in small batches and upsert immediately; never accumulates in RAM."""
    n = len(docs)
    n_batches = (n + embed_batch_size - 1) // embed_batch_size
    for batch_idx, i in enumerate(range(0, n, embed_batch_size), 1):
        b_docs  = docs[i:i + embed_batch_size]
        b_ids   = ids[i:i + embed_batch_size]
        b_metas = metas[i:i + embed_batch_size]
        embeddings = model.encode(b_docs, normalize_embeddings=True, batch_size=embed_batch_size)
        collection.upsert(ids=b_ids, documents=b_docs, embeddings=embeddings.tolist(), metadatas=b_metas)
        del embeddings
        if device == "cuda":
            torch.cuda.empty_cache()
        if n_batches > 1:
            log(f"      batch {batch_idx}/{n_batches}  ({min(i + embed_batch_size, n)}/{n} chunks)")


def index_file_chunks(ids, docs, metas, existing_meta, seen_ids,
                      model, device, embed_batch, collection):
    """Incrementally index one file's chunks against the existing index.

    Records every chunk ID in ``seen_ids`` (used afterwards to prune stale
    chunks). Chunk IDs are content-hashed, so for each chunk:
      - new ID                       -> embed + upsert
      - existing ID, metadata changed -> refresh metadata only (no re-embed)
      - existing ID, metadata same    -> skip

    Embeddings depend only on the chunk body, so a metadata-only edit (e.g. a
    note's frontmatter or a heading) is applied with collection.update without
    paying to re-embed. Returns (n_new, n_updated, n_total)."""
    seen_ids.update(ids)
    new_i = [k for k, cid in enumerate(ids) if cid not in existing_meta]
    upd_i = [k for k, cid in enumerate(ids)
             if cid in existing_meta and metas[k] != existing_meta[cid]]

    if new_i:
        embed_and_upsert(model, device,
                         [docs[k] for k in new_i], [ids[k] for k in new_i],
                         [metas[k] for k in new_i], embed_batch, collection)
    if upd_i:
        collection.update(ids=[ids[k] for k in upd_i],
                          metadatas=[metas[k] for k in upd_i])
    return len(new_i), len(upd_i), len(ids)


def preserve_existing(path_value, existing_meta, seen_ids):
    """Mark a source's already-indexed chunks as seen so the stale-prune step
    does not delete good data when that file's extraction fails this run."""
    seen_ids.update(cid for cid, m in existing_meta.items() if m.get("path") == path_value)


def _index_status(n_new, n_upd, n_total):
    if n_new or n_upd:
        return f"{n_new} new, {n_upd} meta / {n_total} chunks"
    return f"unchanged, {n_total} chunks"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    config = load_config()
    setup_logging(config)

    vault_path      = Path(config["vault_path"]).expanduser().resolve()
    index_path      = config.get("index_path", "./chroma_db")
    collection_name = config.get("collection_name", "obsidian_markdown")
    max_chars       = int(config.get("chunk_max_chars", 1200))
    overlap         = int(config.get("chunk_overlap_chars", 150))
    model_name      = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    embed_batch     = int(config.get("embedding_batch_size", 16))
    md_workers      = int(config.get("markdown_workers", 1))
    pdf_workers     = int(config.get("pdf_workers", 1))

    if not vault_path.exists():
        raise RuntimeError(f"Vault path does not exist: {vault_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    log(f"Vault: {vault_path}")
    log(f"Embedding model: {model_name}  |  device: {device}" +
        (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    log(f"Chunk max chars: {max_chars}  |  overlap: {overlap}")
    log(f"Embed batch size: {embed_batch}  |  md_workers: {md_workers}  |  pdf_workers: {pdf_workers}")

    log("Loading embedding model...")
    model = SentenceTransformer(model_name, device=device)
    log("Model loaded.")

    client = chromadb.PersistentClient(
        path=index_path,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(name=collection_name)

    # Incremental indexing: snapshot the IDs + metadata already in the index.
    # Chunk IDs are content-hashed, so unchanged body text keeps the same ID;
    # each chunk is then embedded (new), metadata-refreshed (same body, changed
    # metadata), or skipped (identical). Every ID seen this run is recorded so
    # leftovers (edited or deleted sources) can be pruned at the end.
    _snap = collection.get(include=["metadatas"])
    existing_meta = dict(zip(_snap["ids"], _snap["metadatas"]))
    existing_ids = set(existing_meta)
    seen_ids = set()
    log(f"Existing chunks in index: {len(existing_ids)}")

    total_chunks = 0     # chunks across successfully-extracted files this run
    total_new = 0        # chunks embedded this run
    total_updated = 0    # chunks whose metadata was refreshed (no re-embed)

    # --- Markdown ---
    md_files = [f for f in sorted(vault_path.rglob("*.md")) if not should_exclude(f, vault_path, config)]
    log(f"\nMarkdown files to index: {len(md_files)}")

    with ThreadPoolExecutor(max_workers=md_workers) as pool:
        futures = [(pool.submit(extract_md_file, f, vault_path, config, max_chars, overlap), f) for f in md_files]
        for file_idx, (future, md_file) in enumerate(futures, 1):
            ids, docs, metas, err = future.result()
            if err:
                log(f"  [{file_idx}/{len(md_files)}] SKIP {md_file.name}: {err.split(':', 1)[-1].strip()}")
                preserve_existing(md_file.relative_to(vault_path).as_posix(), existing_meta, seen_ids)
                continue
            if not docs:
                continue
            n_new, n_upd, n_total = index_file_chunks(ids, docs, metas, existing_meta, seen_ids,
                                                      model, device, embed_batch, collection)
            log(f"  [{file_idx}/{len(md_files)}] {md_file.name}  ({_index_status(n_new, n_upd, n_total)})")
            total_chunks  += n_total
            total_new     += n_new
            total_updated += n_upd
            del ids, docs, metas
            gc.collect()

    log(f"Markdown complete: {total_chunks} chunks ({total_new} embedded this run)")

    # --- PDF sources ---
    for pdf_source in config.get("pdf_sources", []):
        pdf_dir     = Path(pdf_source["path"]).expanduser().resolve()
        source_type = pdf_source.get("type", "resource")

        if not pdf_dir.exists():
            log(f"Warning: pdf_source path does not exist, skipping: {pdf_dir}")
            continue

        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        log(f"\nPDF source [{source_type}]: {len(pdf_files)} files — {pdf_dir.name}")
        source_chunks = 0
        source_new = 0

        with ThreadPoolExecutor(max_workers=pdf_workers) as pool:
            futures = [(pool.submit(extract_pdf_file, f, source_type, max_chars, overlap), f) for f in pdf_files]
            for file_idx, (future, pdf_file) in enumerate(futures, 1):
                ids, docs, metas, err = future.result()
                if err:
                    log(f"  [{file_idx}/{len(pdf_files)}] SKIP {pdf_file.name}: {err.split(':', 1)[-1].strip()}")
                    preserve_existing(pdf_file.name, existing_meta, seen_ids)
                    continue
                if not docs:
                    continue
                n_new, n_upd, n_total = index_file_chunks(ids, docs, metas, existing_meta, seen_ids,
                                                          model, device, embed_batch, collection)
                log(f"  [{file_idx}/{len(pdf_files)}] {pdf_file.name}  ({_index_status(n_new, n_upd, n_total)})")
                total_chunks  += n_total
                total_new     += n_new
                total_updated += n_upd
                source_chunks += n_total
                source_new    += n_new
                del ids, docs, metas
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

        log(f"  [{source_type}] complete: {source_chunks} chunks ({source_new} embedded this run)")

    # --- Pre-extracted document JSON (books/resources from doc-text-extractor) ---
    for json_source in config.get("json_sources", []):
        json_dir = Path(json_source["path"]).expanduser().resolve()

        if not json_dir.exists():
            log(f"Warning: json_source path does not exist, skipping: {json_dir}")
            continue

        json_files = sorted(json_dir.glob("*.json"))
        log(f"\nJSON source: {len(json_files)} files — {json_dir.name}")
        source_chunks = 0
        source_new = 0

        with ThreadPoolExecutor(max_workers=pdf_workers) as pool:
            futures = [(pool.submit(extract_json_doc, f, max_chars, overlap), f) for f in json_files]
            for file_idx, (future, json_file) in enumerate(futures, 1):
                ids, docs, metas, err = future.result()
                if err:
                    # A JSON that fails to parse can't be mapped back to its stored
                    # chunks (their metadata path is the source file_name, not the
                    # .json filename), so skip without preserving — a corrupt JSON's
                    # chunks then fall through to the stale prune.
                    log(f"  [{file_idx}/{len(json_files)}] SKIP {json_file.name}: {err.split(':', 1)[-1].strip()}")
                    continue
                if not docs:
                    continue
                n_new, n_upd, n_total = index_file_chunks(ids, docs, metas, existing_meta, seen_ids,
                                                          model, device, embed_batch, collection)
                log(f"  [{file_idx}/{len(json_files)}] {json_file.name}  ({_index_status(n_new, n_upd, n_total)})")
                total_chunks  += n_total
                total_new     += n_new
                total_updated += n_upd
                source_chunks += n_total
                source_new    += n_new
                del ids, docs, metas
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

        log(f"  [json] complete: {source_chunks} chunks ({source_new} embedded this run)")

    # Prune chunks that no longer exist in any source (edited or deleted files).
    stale = existing_ids - seen_ids
    if stale:
        stale_list = list(stale)
        for i in range(0, len(stale_list), 500):
            collection.delete(ids=stale_list[i:i + 500])
        log(f"Removed {len(stale)} stale chunks (edited or deleted sources)")

    log(f"\nIndexing complete. Index now holds {len(seen_ids)} chunks "
        f"({total_new} embedded, {total_updated} metadata-updated, "
        f"{total_chunks - total_new - total_updated} unchanged, {len(stale)} removed).")


if __name__ == "__main__":
    main()
