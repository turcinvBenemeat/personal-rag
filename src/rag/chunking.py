"""Pure text helpers shared by every extractor: heading split, char chunking,
wikilink extraction, and content-hashed chunk IDs. No I/O, no embedding."""

import hashlib
import re


def extract_wikilinks(text: str):
    return sorted(set(re.findall(r"\[\[([^\]|#]+)", text)))


# Vault convention: every note ends with navigation-only sections ("# Related
# Topics" and "## Potential New Notes") that are pure wikilink lists. They carry
# graph structure (captured separately via extract_wikilinks) but no semantic
# content — embedding them dilutes retrieval with title soup (~16% of corpus).
_NAV_TAIL = re.compile(r"(?m)^#{1,6}\s*(Related Topics|Potential New Notes)\s*$")


def strip_navigation_tail(text: str) -> str:
    """Drop everything from the first navigation heading to the end of the note."""
    m = _NAV_TAIL.search(text)
    return text[: m.start()].rstrip() if m else text


def strip_wikilink_syntax(text: str) -> str:
    """Inline [[target|alias]] -> alias, [[target]] -> target.

    Embedding models see plain words instead of bracket noise; the link graph
    itself is preserved in chunk metadata by extract_wikilinks (run it first).
    """
    text = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r"\1", text)
    return re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)


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
    """Deterministic chunk ID. Hashing the full chunk text makes identical
    content yield the same ID (the basis for incremental/idempotent indexing)."""
    raw = "::".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
