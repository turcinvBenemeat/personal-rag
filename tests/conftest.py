# test_queries.py is a manual integration smoke test — it needs a populated
# ChromaDB index and loads the embedding model, so it is not part of the offline
# unit suite. Keep pytest from collecting (and importing) it.
collect_ignore = ["test_queries.py"]
