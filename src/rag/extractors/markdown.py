"""Obsidian Markdown extractor — frontmatter + heading-aware chunking."""

import re
from pathlib import Path

import frontmatter

from ..chunking import chunk_text, extract_wikilinks, split_by_headings, stable_id


def should_exclude(path: Path, vault_path: Path, config: dict) -> bool:
    rel = path.relative_to(vault_path).as_posix()
    for excluded in config.get("exclude_dirs", []):
        excluded = excluded.strip("/")
        if rel == excluded or rel.startswith(excluded + "/"):
            return True
    if path.name in config.get("exclude_files", []):
        return True
    return False


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
