.PHONY: help index query test build build-jetson \
        docker-index docker-query docker-test \
        jetson-index jetson-query jetson-test

PYTHON := .venv/bin/python
Q      ?=
K      ?=

help:
	@echo ""
	@echo "Local (venv):"
	@echo "  make index              reindex vault + PDFs"
	@echo "  make query Q=\"...\"      semantic query"
	@echo "  make test [K=keyword]   smoke tests (optional keyword filter)"
	@echo ""
	@echo "Docker x86:"
	@echo "  make build              build personal-rag:latest"
	@echo "  make docker-index       reindex inside container"
	@echo "  make docker-query Q=\"...\""
	@echo "  make docker-test [K=keyword]"
	@echo ""
	@echo "Docker Jetson (run on Jetson):"
	@echo "  make build-jetson       build personal-rag:jetson"
	@echo "  make jetson-index"
	@echo "  make jetson-query Q=\"...\""
	@echo "  make jetson-test [K=keyword]"
	@echo ""

# ── Local ─────────────────────────────────────────────────────────────────────

index:
	$(PYTHON) index_obsidian.py

query:
	$(PYTHON) query_obsidian.py $(Q)

test:
	$(PYTHON) test_queries.py $(K)

# ── Docker x86 ────────────────────────────────────────────────────────────────

build:
	docker build -t personal-rag:latest .

docker-index:
	docker compose run --rm rag python index_obsidian.py

docker-query:
	docker compose run --rm rag python query_obsidian.py $(Q)

docker-test:
	docker compose run --rm rag python test_queries.py $(K)

# ── Docker Jetson ─────────────────────────────────────────────────────────────

build-jetson:
	docker build -f Dockerfile.jetson -t personal-rag:jetson .

jetson-index:
	docker compose -f docker-compose.jetson.yml run --rm rag python index_obsidian.py

jetson-query:
	docker compose -f docker-compose.jetson.yml run --rm rag python query_obsidian.py $(Q)

jetson-test:
	docker compose -f docker-compose.jetson.yml run --rm rag python test_queries.py $(K)
