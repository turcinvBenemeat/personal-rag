"""
Shared utilities for the rag package.

Imported before chromadb in every entry point — module-level code suppresses
ChromaDB telemetry noise at startup.
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


def _find_config() -> Path:
    """Locate config.yaml: RAG_CONFIG_PATH env var → cwd → package root."""
    if cfg_env := os.environ.get("RAG_CONFIG_PATH"):
        return Path(cfg_env)
    cwd_cfg = Path.cwd() / "config.yaml"
    if cwd_cfg.exists():
        return cwd_cfg
    # Editable install: src/rag/utils.py → ../../../ = project root
    return Path(__file__).resolve().parent.parent.parent / "config.yaml"


def load_config() -> dict:
    """Load config.yaml and apply environment variable overrides.

    Override user-specific paths without editing config.yaml:
        RAG_VAULT_PATH          overrides vault_path
        RAG_PDF_BOOKS_PATH      overrides pdf_sources entry with type=book
        RAG_PDF_RESOURCES_PATH  overrides pdf_sources entry with type=resource
        RAG_INDEX_PATH          overrides index_path (ChromaDB storage dir)
        RAG_CONFIG_PATH         overrides the config.yaml location itself
    """
    config_path = _find_config()
    with open(config_path, "r", encoding="utf-8") as f:
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
