.PHONY: help install index query test test-unit build build-jetson \
        docker-index docker-query docker-test \
        jetson-index jetson-query jetson-test

PYTHON := .venv/bin/python
Q      ?=
K      ?=

help:
	@echo ""
	@echo "Setup:"
	@echo "  make install            create venv and install package"
	@echo ""
	@echo "Local:"
	@echo "  make index              reindex vault + PDFs"
	@echo "  make query Q=\"...\"      semantic query"
	@echo "  make test-unit          offline pytest unit suite"
	@echo "  make test [K=keyword]   retrieval smoke tests (needs an index)"
	@echo ""
	@echo "Docker x86:"
	@echo "  make build              build personal-rag:latest"
	@echo "  make docker-index"
	@echo "  make docker-query Q=\"...\""
	@echo "  make docker-test [K=keyword]"
	@echo ""
	@echo "Docker Jetson (run on Jetson):"
	@echo "  make build-jetson       build personal-rag:jetson"
	@echo "  make jetson-index"
	@echo "  make jetson-query Q=\"...\""
	@echo "  make jetson-test [K=keyword]"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────

install:
	uv venv .venv
	uv pip install -r requirements.txt
	uv pip install -e . --no-deps

# ── Local ─────────────────────────────────────────────────────────────────────

index:
	.venv/bin/rag-index

query:
	.venv/bin/rag-query $(Q)

test:
	$(PYTHON) tests/test_queries.py $(K)

test-unit:
	$(PYTHON) -m pytest tests/ -q

# ── Docker x86 ────────────────────────────────────────────────────────────────

build:
	docker build -t personal-rag:latest .

docker-index:
	docker compose run --rm rag python -m rag.indexer

docker-query:
	docker compose run --rm rag python -m rag.query $(Q)

docker-test:
	docker compose run --rm rag python tests/test_queries.py $(K)

# ── Docker Jetson ─────────────────────────────────────────────────────────────

build-jetson:
	docker build -f Dockerfile.jetson -t personal-rag:jetson .

jetson-index:
	docker compose -f docker-compose.jetson.yml run --rm rag python -m rag.indexer

jetson-query:
	docker compose -f docker-compose.jetson.yml run --rm rag python -m rag.query $(Q)

jetson-test:
	docker compose -f docker-compose.jetson.yml run --rm rag python tests/test_queries.py $(K)
