"""Unit tests for the incremental engine (rag.indexing) — new/update/skip,
stale preservation, and end-to-end idempotency through the source registry.

Uses a fake embedding model and a temporary on-disk ChromaDB, so it runs
offline with no GPU and no real index."""

import tempfile

import chromadb
import numpy as np

from rag.extractors import iter_sources
from rag.indexing import index_file_chunks, preserve_existing, run_source


class FakeModel:
    """Stand-in for SentenceTransformer — shape-correct, content-irrelevant."""
    def encode(self, docs, normalize_embeddings=True, batch_size=16):
        return np.zeros((len(docs), 4))


def _collection(name="unit"):
    client = chromadb.PersistentClient(
        path=tempfile.mkdtemp(),
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(name)


def test_index_file_chunks_new_update_skip():
    col = _collection("nus")
    model = FakeModel()
    ids = ["i1", "i2"]
    docs = ["alpha", "beta"]
    metas = [{"path": "p", "domain": "DevOps"}, {"path": "p", "domain": "DevOps"}]

    # first pass: both new -> embedded
    assert index_file_chunks(ids, docs, metas, {}, set(), model, "cpu", 16, col) == (2, 0, 2)

    snap = col.get(include=["metadatas"])
    existing = dict(zip(snap["ids"], snap["metadatas"]))

    # identical content + metadata -> skipped
    assert index_file_chunks(ids, docs, metas, existing, set(), model, "cpu", 16, col) == (0, 0, 2)

    # same body, changed metadata -> metadata refresh only (no re-embed)
    changed = [{"path": "p", "domain": "Platform"}, {"path": "p", "domain": "Platform"}]
    assert index_file_chunks(ids, docs, changed, existing, set(), model, "cpu", 16, col) == (0, 2, 2)


def test_preserve_existing_marks_only_matching_path():
    existing = {"a": {"path": "x"}, "b": {"path": "y"}, "c": {"path": "x"}}
    seen = set()
    preserve_existing("x", existing, seen)
    assert seen == {"a", "c"}


def test_run_source_end_to_end_is_idempotent(tmp_path):
    (tmp_path / "Knowledge").mkdir()
    (tmp_path / "Knowledge" / "n.md").write_text(
        "---\ntitle: N\ndomain: DevOps\n---\n# H\n" + "word " * 300, encoding="utf-8")
    (tmp_path / "Templates").mkdir()
    (tmp_path / "Templates" / "t.md").write_text("---\ntitle: T\n---\nignored " * 50, encoding="utf-8")

    config = {
        "vault_path": str(tmp_path), "exclude_dirs": ["Templates"], "exclude_files": [],
        "markdown_workers": 1, "pdf_workers": 1,
    }
    col = _collection("e2e")
    model = FakeModel()

    def run():
        snap = col.get(include=["metadatas"])
        existing = dict(zip(snap["ids"], snap["metadatas"]))
        seen = set()
        new = upd = 0
        for source in iter_sources(config, tmp_path, 1200, 150):
            _, s_new, s_upd = run_source(source, existing, seen, model, "cpu", 16, col)
            new += s_new
            upd += s_upd
        return new, upd, len(set(existing) - seen)

    first = run()
    second = run()
    assert first[0] > 0          # first run embeds the (non-excluded) note
    assert second == (0, 0, 0)   # nothing changed -> no work, nothing pruned
