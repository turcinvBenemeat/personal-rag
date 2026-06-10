"""Unit tests for the pure text helpers in rag.chunking (no I/O, no deps)."""

from rag.chunking import (
    chunk_text,
    extract_wikilinks,
    split_by_headings,
    stable_id,
    strip_navigation_tail,
    strip_wikilink_syntax,
)


def test_chunk_text_short_returns_single():
    assert chunk_text("hello world", 1200, 150) == ["hello world"]


def test_chunk_text_long_windows_with_overlap():
    chunks = chunk_text("a" * 3000, 1000, 100)
    assert len(chunks) >= 3
    assert all(len(c) <= 1000 for c in chunks)
    # overlap means the chunks together cover more than the original length
    assert sum(len(c) for c in chunks) > 3000


def test_chunk_text_collapses_blank_lines():
    assert chunk_text("a\n\n\n\nb", 1200, 150) == ["a\n\nb"]


def test_stable_id_is_deterministic():
    assert stable_id("path", 0, 0, "body") == stable_id("path", 0, 0, "body")


def test_stable_id_is_content_and_position_sensitive():
    base = stable_id("path", 0, 0, "a")
    assert base != stable_id("path", 0, 0, "b")   # content
    assert base != stable_id("path", 0, 1, "a")   # position
    assert base != stable_id("other", 0, 0, "a")  # source


def test_split_by_headings_groups_body_under_heading():
    assert split_by_headings("# A\nalpha\n# B\nbeta") == [("A", "alpha"), ("B", "beta")]


def test_split_by_headings_drops_empty_sections():
    secs = split_by_headings("# Empty\n\n# Real\ncontent")
    assert secs == [("Real", "content")]


def test_extract_wikilinks_dedups_sorts_and_strips_alias_anchor():
    text = "see [[K3s]] and [[Docker|containers]] and [[K3s#setup]] and [[K3s]]"
    assert extract_wikilinks(text) == ["Docker", "K3s"]


def test_strip_navigation_tail_cuts_related_topics_to_end():
    body = "# Summary\nreal content\n# Related Topics\n- [[A]]\n## Potential New Notes\n- [[B]]"
    assert strip_navigation_tail(body) == "# Summary\nreal content"


def test_strip_navigation_tail_cuts_lone_potential_new_notes():
    body = "# Summary\ncontent\n## Potential New Notes\n- [[X]]"
    assert strip_navigation_tail(body) == "# Summary\ncontent"


def test_strip_navigation_tail_noop_without_nav_headings():
    body = "# Summary\nmentions Related Topics inline, not as heading"
    assert strip_navigation_tail(body) == body


def test_strip_wikilink_syntax_plain_and_alias():
    text = "use [[K3s]] with [[Books/docker.pdf|Docker book]] today"
    assert strip_wikilink_syntax(text) == "use K3s with Docker book today"
