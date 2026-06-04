import hashlib
import os
import re
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import posthog as _posthog
_posthog.capture = lambda *a, **kw: None  # chromadb 0.6.x / posthog 7.x signature mismatch

import chromadb
import frontmatter
import yaml
from sentence_transformers import SentenceTransformer

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


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

    print(f"Vault: {vault_path}")
    print(f"Embedding model: {model_name}")

    model = SentenceTransformer(model_name)

    client = chromadb.PersistentClient(
        path=index_path,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )

    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass

    collection = client.create_collection(name=collection_name)

    ids = []
    documents = []
    metadatas = []

    md_files = sorted(vault_path.rglob("*.md"))

    for md_file in md_files:
        if should_exclude(md_file, vault_path, config):
            continue

        rel_path = md_file.relative_to(vault_path).as_posix()

        try:
            raw = md_file.read_text(encoding="utf-8", errors="ignore")
            raw = re.sub(r"\{\{[^}]+\}\}", "", raw)  # strip unresolved Obsidian template vars
            parsed = frontmatter.loads(raw)
            body = parsed.content.strip()
            meta = dict(parsed.metadata)
        except Exception as exc:
            print(f"skip parse error: {rel_path}: {exc}")
            continue

        if not body:
            continue

        title = str(meta.get("title") or md_file.stem)
        note_type = str(meta.get("type") or "")
        domain = str(meta.get("domain") or "")
        status = str(meta.get("status") or "")
        source = str(meta.get("source") or "")

        tags_value = meta.get("tags") or []
        if isinstance(tags_value, list):
            tags = ", ".join(str(t) for t in tags_value)
        else:
            tags = str(tags_value)

        wikilinks = ", ".join(extract_wikilinks(body))

        sections = split_by_headings(body)

        for section_index, (heading, section_text) in enumerate(sections):
            chunks = chunk_text(section_text, max_chars=max_chars, overlap=overlap)

            for chunk_index, chunk in enumerate(chunks):
                chunk_id = stable_id(rel_path, section_index, chunk_index, chunk[:80])

                ids.append(chunk_id)
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

    print(f"Markdown chunks: {len(documents)}")

    if not documents:
        raise RuntimeError("No documents found to index.")

    batch_size = 128

    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        batch_docs = documents[start:end]
        batch_embeddings = model.encode(batch_docs, normalize_embeddings=True).tolist()

        collection.add(
            ids=ids[start:end],
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=metadatas[start:end],
        )

        print(f"Indexed {min(end, len(documents))}/{len(documents)} chunks")

    print("Indexing complete.")


if __name__ == "__main__":
    main()
