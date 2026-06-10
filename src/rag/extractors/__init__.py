"""Source registry.

Each indexing source (Markdown vault, PDF dirs, JSON dirs) is described by a
``Source`` whose ``extract`` callable returns the common
``(ids, documents, metadatas, error)`` contract. ``iter_sources()`` turns a
config into that uniform stream so the indexing engine can process them all with
one loop. Adding a new source type = a new module here + a block in
``iter_sources`` + a ``config.yaml`` entry; ``indexer.main()`` is untouched.
"""

from collections import namedtuple
from pathlib import Path

from .json_doc import extract_json_doc
from .markdown import extract_md_file, should_exclude
from .pdf import extract_pdf_file

__all__ = ["Source", "iter_sources", "extract_md_file", "extract_pdf_file",
           "extract_json_doc", "should_exclude"]

# kind/label: for logging. files: list[Path]. workers: thread count.
# extract(file) -> (ids, docs, metas, err). preserve_key(file) -> path string,
# or None when a failed extraction can't be mapped back to its stored chunks.
Source = namedtuple("Source", "kind label files workers extract preserve_key")


def _dir_source(kind, src, glob, workers, make_extract, make_preserve):
    """Build a Source for one configured directory entry (PDF / JSON)."""
    d = Path(src["path"]).expanduser().resolve()
    if not d.exists():
        return Source(kind, f"{kind.upper()} source: path does not exist, skipped — {d}",
                      [], workers, None, None)
    files = sorted(d.glob(glob))
    label = f"{kind.upper()} source [{src.get('type')}]" if src.get("type") else f"{kind.upper()} source"
    return Source(kind, f"{label}: {len(files)} files — {d.name}", files, workers,
                  make_extract(src), make_preserve)


def _json_covered_filenames(config: dict) -> set:
    """``file_name`` of every document available as pre-extracted JSON.

    These JSONs carry full text + enriched metadata, so they are always
    preferred over parsing the original file live; PDF sources then act as a
    fallback for files the extraction pipeline has not covered yet.
    """
    import json as _json

    covered = set()
    for src in config.get("json_sources", []):
        d = Path(src["path"]).expanduser().resolve()
        if not d.exists():
            continue
        for p in d.glob("*.json"):
            try:
                name = _json.loads(p.read_text(encoding="utf-8")).get("file_name")
            except Exception:
                continue
            if name:
                covered.add(name)
    return covered


def iter_sources(config: dict, vault_path: Path, max_chars: int, overlap: int):
    """Yield a ``Source`` for the Markdown vault and every configured PDF/JSON dir.

    PDF sources are a *fallback*: any file whose name is already covered by a
    ``json_sources`` document is skipped, so books/resources are indexed exactly
    once (from the richer pre-extracted JSON) and live PDF parsing only handles
    new files the doc-text-extractor pipeline hasn't processed yet.
    """
    md_workers = int(config.get("markdown_workers", 1))
    pdf_workers = int(config.get("pdf_workers", 1))

    # Markdown vault (single source)
    md_files = [f for f in sorted(vault_path.rglob("*.md")) if not should_exclude(f, vault_path, config)]
    yield Source(
        "markdown", f"Markdown: {len(md_files)} files", md_files, md_workers,
        lambda f: extract_md_file(f, vault_path, config, max_chars, overlap),
        lambda f: f.relative_to(vault_path).as_posix(),
    )

    json_covered = _json_covered_filenames(config)

    # PDF source directories (fallback for files without pre-extracted JSON)
    for src in config.get("pdf_sources", []):
        source = _dir_source(
            "pdf", src, "*.pdf", pdf_workers,
            lambda s: (lambda f, t=s.get("type", "resource"): extract_pdf_file(f, t, max_chars, overlap)),
            lambda f: f.name,
        )
        if json_covered and source.files:
            kept = [f for f in source.files if f.name not in json_covered]
            skipped = len(source.files) - len(kept)
            label = source.label + (f" ({skipped} covered by JSON, skipped)" if skipped else "")
            source = source._replace(files=kept, label=label)
        yield source

    # Pre-extracted document JSON directories
    for src in config.get("json_sources", []):
        yield _dir_source(
            "json", src, "*.json", pdf_workers,
            lambda s: (lambda f: extract_json_doc(f, max_chars, overlap)),
            None,  # a corrupt JSON can't be mapped to its file_name → no preserve
        )
