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


def iter_sources(config: dict, vault_path: Path, max_chars: int, overlap: int):
    """Yield a ``Source`` for the Markdown vault and every configured PDF/JSON dir."""
    md_workers = int(config.get("markdown_workers", 1))
    pdf_workers = int(config.get("pdf_workers", 1))

    # Markdown vault (single source)
    md_files = [f for f in sorted(vault_path.rglob("*.md")) if not should_exclude(f, vault_path, config)]
    yield Source(
        "markdown", f"Markdown: {len(md_files)} files", md_files, md_workers,
        lambda f: extract_md_file(f, vault_path, config, max_chars, overlap),
        lambda f: f.relative_to(vault_path).as_posix(),
    )

    # PDF source directories
    for src in config.get("pdf_sources", []):
        stype = src.get("type", "resource")
        yield _dir_source(
            "pdf", src, "*.pdf", pdf_workers,
            lambda s: (lambda f, t=s.get("type", "resource"): extract_pdf_file(f, t, max_chars, overlap)),
            lambda f: f.name,
        )

    # Pre-extracted document JSON directories
    for src in config.get("json_sources", []):
        yield _dir_source(
            "json", src, "*.json", pdf_workers,
            lambda s: (lambda f: extract_json_doc(f, max_chars, overlap)),
            None,  # a corrupt JSON can't be mapped to its file_name → no preserve
        )
