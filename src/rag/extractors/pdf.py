"""PDF extractor — page text into char-window chunks (live parsing via pypdf)."""

import re
from pathlib import Path

from pypdf import PdfReader

from ..chunking import chunk_text, stable_id


def clean_pdf_title(filename: str) -> str:
    """Fallback title from filename when PDF metadata is unavailable."""
    name = Path(filename).stem
    name = re.sub(r"[_\-]?(v\d+|[23456]e|[23456]rdedition|[23456]thedition)$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[_\-]+", " ", name)
    return name.strip().title()


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
