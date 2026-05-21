# environment setup

.PHONY: setup install clean

setup:
	uv venv
	. .venv/bin/activate && uv pip install -e .

install:
	uv pip install -e .

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__

# development

.PHONY: lint format test check

lint:
	ruff check .

format:
	black .

test:
	pytest

check: lint test

# Qdrant 

.PHONY: qdrant-up qdrant-down qdrant-logs

qdrant-up:
	docker compose up -d qdrant

qdrant-down:
	docker compose down

qdrant-logs:
	docker compose logs -f qdrant

# retrieval 

.PHONY: index ingest bm25 dense hybrid agentic evaluate full

# Build / refresh Qdrant index
index:
	python -m src.experiments.setup_index

# Optional explicit ingestion step
ingest:
	python -m src.ingest.build_corpus

# Individual retrieval evaluations
bm25:
	python -m src.experiments.run_bm25

dense:
	python -m src.experiments.run_dense

hybrid:
	python -m src.experiments.run_hybrid

agentic:
	python -m src.experiments.run_agentic

# Unified evaluation runner
evaluate:
	python -m src.experiments.run_all

# Full end-to-end pipeline:
# ingestion -> indexing -> evaluation
full: qdrant-up ingest index evaluate