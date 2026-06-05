"""
Shared utilities for personal-rag scripts.

Import this module BEFORE importing chromadb in any script — the module-level
code suppresses ChromaDB telemetry noise at startup.
"""

import os
from pathlib import Path

# Suppress ChromaDB/posthog telemetry before chromadb is imported anywhere.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import posthog as _posthog
_posthog.capture = lambda *a, **kw: None  # chromadb 0.6.x / posthog 7.x signature mismatch

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    """Load config.yaml and apply environment variable overrides for user-specific paths.

    Override any path without editing config.yaml:
        RAG_VAULT_PATH          overrides vault_path
        RAG_PDF_BOOKS_PATH      overrides the pdf_sources entry with type=book
        RAG_PDF_RESOURCES_PATH  overrides the pdf_sources entry with type=resource
    """
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if vault := os.environ.get("RAG_VAULT_PATH"):
        cfg["vault_path"] = vault
    if books := os.environ.get("RAG_PDF_BOOKS_PATH"):
        for src in cfg.get("pdf_sources", []):
            if src.get("type") == "book":
                src["path"] = books
    if resources := os.environ.get("RAG_PDF_RESOURCES_PATH"):
        for src in cfg.get("pdf_sources", []):
            if src.get("type") == "resource":
                src["path"] = resources

    if index := os.environ.get("RAG_INDEX_PATH"):
        cfg["index_path"] = index

    return cfg
