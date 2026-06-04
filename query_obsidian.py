import os
import sys
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import posthog as _posthog
_posthog.capture = lambda *a, **kw: None  # chromadb 0.6.x / posthog 7.x signature mismatch

import chromadb
import yaml
from sentence_transformers import SentenceTransformer

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config():
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    if len(sys.argv) < 2:
        print("Usage: python query_obsidian.py \"your question\" [n_results]")
        raise SystemExit(1)

    # Last arg is treated as n_results if it's a plain integer
    args = sys.argv[1:]
    if len(args) >= 2 and args[-1].isdigit():
        n_results = int(args[-1])
        query = " ".join(args[:-1])
    else:
        n_results = 8
        query = " ".join(args)

    config = load_config()
    model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    index_path = config.get("index_path", "./chroma_db")
    collection_name = config.get("collection_name", "obsidian_markdown")

    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()[0]

    client = chromadb.PersistentClient(
        path=index_path,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(collection_name)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    print()
    print("Query: " + query)
    print()

    for i, item in enumerate(zip(docs, metas, distances), start=1):
        doc, meta, distance = item

        print("=" * 80)
        print(str(i) + ". " + str(meta.get("title")) + " - " + str(meta.get("heading")))
        print("Path: " + str(meta.get("path")))
        print(
            "Type: "
            + str(meta.get("type"))
            + " | Domain: "
            + str(meta.get("domain"))
            + " | Status: "
            + str(meta.get("status"))
        )
        print("Distance: " + format(distance, ".4f"))
        print("-" * 80)
        print(doc[:1200].strip())
        print()


if __name__ == "__main__":
    main()
