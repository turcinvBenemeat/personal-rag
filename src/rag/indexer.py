"""Streaming indexer — Obsidian Markdown + PDF → ChromaDB."""

import gc
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .utils import load_config  # sets telemetry env var and patches posthog before chromadb loads

import torch
import chromadb
import frontmatter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


def log(msg: str) -> None:
    """Print with immediate flush so progress is visible even when piped."""
    print(msg, flush=True)


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
                "path": rel_path, "title": title, "heading": heading,
                "type": note_type, "domain": domain, "status": status,
                "source": source, "tags": tags, "wikilinks": wikilinks,
            })
    return ids, documents, metadatas, None


def extract_pdf_file(pdf_path: Path, source_type: str, max_chars: int, overlap: int):
    """Extract text from one PDF → (ids, documents, metadatas, error)."""
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
                "path": pdf_path.name, "title": title, "heading": heading,
                "type": source_type, "domain": "", "status": "",
                "source": "pdf", "tags": "", "wikilinks": "",
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
        collection.add(ids=b_ids, documents=b_docs, embeddings=embeddings.tolist(), metadatas=b_metas)
        del embeddings
        if device == "cuda":
            torch.cuda.empty_cache()
        if n_batches > 1:
            log(f"      batch {batch_idx}/{n_batches}  ({min(i + embed_batch_size, n)}/{n} chunks)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    config = load_config()

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
    try:
        client.delete_collection(collection_name)
        log(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass

    collection = client.create_collection(name=collection_name)
    total_chunks = 0

    # --- Markdown ---
    md_files = [f for f in sorted(vault_path.rglob("*.md")) if not should_exclude(f, vault_path, config)]
    log(f"\nMarkdown files to index: {len(md_files)}")

    with ThreadPoolExecutor(max_workers=md_workers) as pool:
        futures = [(pool.submit(extract_md_file, f, vault_path, config, max_chars, overlap), f) for f in md_files]
        for file_idx, (future, md_file) in enumerate(futures, 1):
            ids, docs, metas, err = future.result()
            if err:
                log(f"  [{file_idx}/{len(md_files)}] SKIP {md_file.name}: {err.split(':', 1)[-1].strip()}")
                continue
            if not docs:
                continue
            log(f"  [{file_idx}/{len(md_files)}] {md_file.name}  ({len(docs)} chunks)")
            embed_and_upsert(model, device, docs, ids, metas, embed_batch, collection)
            total_chunks += len(docs)
            del ids, docs, metas
            gc.collect()

    log(f"Markdown complete: {total_chunks} chunks indexed")

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

        with ThreadPoolExecutor(max_workers=pdf_workers) as pool:
            futures = [(pool.submit(extract_pdf_file, f, source_type, max_chars, overlap), f) for f in pdf_files]
            for file_idx, (future, pdf_file) in enumerate(futures, 1):
                ids, docs, metas, err = future.result()
                if err:
                    log(f"  [{file_idx}/{len(pdf_files)}] SKIP {pdf_file.name}: {err.split(':', 1)[-1].strip()}")
                    continue
                if not docs:
                    continue
                log(f"  [{file_idx}/{len(pdf_files)}] {pdf_file.name}  ({len(docs)} chunks)")
                embed_and_upsert(model, device, docs, ids, metas, embed_batch, collection)
                total_chunks  += len(docs)
                source_chunks += len(docs)
                del ids, docs, metas
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

        log(f"  [{source_type}] complete: {source_chunks} chunks indexed")

    log(f"\nIndexing complete. Total chunks: {total_chunks}")


if __name__ == "__main__":
    main()
