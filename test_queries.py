"""
Retrieval smoke tests for personal-rag.

Run after every reindex to verify query quality across all vault domains.

Usage:
    .venv/bin/python test_queries.py            # run all queries
    .venv/bin/python test_queries.py kubernetes # run queries matching keyword

Each query prints the top 3 results. Distance < 0.6 is a strong match,
0.6-0.8 is acceptable, > 0.8 is weak.
"""

import os
import sys

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import posthog as _posthog
_posthog.capture = lambda *a, **kw: None

import chromadb
import yaml
from pathlib import Path
from sentence_transformers import SentenceTransformer

_CONFIG_PATH = Path(__file__).parent / "config.yaml"

# ---------------------------------------------------------------------------
# Test queries — one or two per domain, plus cross-domain
# Add your own as your vault grows.
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    # DevOps
    ("DevOps",              "What do I know about Kubernetes and container orchestration?"),
    ("DevOps",              "When should I use K3s instead of full Kubernetes?"),
    ("DevOps",              "How do I optimise cloud costs when learning DevOps?"),

    # AI & Automation
    ("AI & Automation",     "What tools and SDKs do I use for AI automation?"),
    ("AI & Automation",     "What have I learned about Gemini SDK and RAG?"),

    # Software Engineering
    ("Software Engineering","How should I handle secrets in Python services?"),
    ("Software Engineering","What are my notes on SQLite schema and migrations?"),
    ("Software Engineering","How do I design SMS notification services?"),

    # Bioprocessing
    ("Bioprocessing",       "What do I know about bioprocessing workflows?"),

    # Business Development
    ("Business Development","What are my notes on business development strategy?"),

    # Career Development
    ("Career Development",  "What skills should I focus on for career growth?"),

    # Cross-domain
    ("Cross-domain",        "What decisions have I documented and why?"),
    ("Cross-domain",        "What frameworks do I use for problem solving?"),
]

N_RESULTS = 3
DISTANCE_WARN = 0.75  # print a warning above this threshold


def load_config():
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_tests(filter_keyword: str = ""):
    config = load_config()
    model = SentenceTransformer(config["embedding_model"])
    client = chromadb.PersistentClient(
        path=config["index_path"],
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(config["collection_name"])

    queries = TEST_QUERIES
    if filter_keyword:
        queries = [(d, q) for d, q in queries if filter_keyword.lower() in q.lower() or filter_keyword.lower() in d.lower()]
        if not queries:
            print(f"No queries match '{filter_keyword}'")
            return

    passed = 0
    warned = 0

    for domain, query in queries:
        embedding = model.encode([query], normalize_embeddings=True).tolist()[0]
        results = collection.query(
            query_embeddings=[embedding],
            n_results=N_RESULTS,
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        best = distances[0] if distances else 1.0

        status = "OK" if best < DISTANCE_WARN else "WEAK"
        if status == "WEAK":
            warned += 1
        else:
            passed += 1

        print()
        print(f"[{domain}] [{status}]  {query}")
        print(f"  Best distance: {best:.4f}")

        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
            title = meta.get("title", "?")
            heading = meta.get("heading", "")
            path = meta.get("path", "")
            heading_str = f" — {heading}" if heading and heading != title else ""
            print(f"  {i}. [{dist:.4f}] {title}{heading_str}")
            print(f"       {path}")

    total = passed + warned
    print()
    print("=" * 60)
    print(f"Results: {passed}/{total} OK  |  {warned}/{total} WEAK (distance >= {DISTANCE_WARN})")
    print("=" * 60)


if __name__ == "__main__":
    keyword = sys.argv[1] if len(sys.argv) > 1 else ""
    run_tests(keyword)
