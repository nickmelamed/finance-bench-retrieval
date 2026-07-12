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

.PHONY: index align bm25 dense hybrid agentic evaluate dashboard full

# Full ingestion (chunk + save) + embed/upload to Qdrant
index:
	python -m finance_bench.experiments.setup_index

# Align gold evidence to chunk ids (run after `index`, before `evaluate` -
# `index` overwrites financebench_examples.json from the raw dataset)
align:
	python -m finance_bench.evaluation.build_gold_alignment

# Individual retrieval evaluations
bm25:
	python -m finance_bench.experiments.run_bm25

dense:
	python -m finance_bench.experiments.run_dense

hybrid:
	python -m finance_bench.experiments.run_hybrid

agentic:
	python -m finance_bench.experiments.run_agentic

# Unified evaluation runner
evaluate:
	python -m finance_bench.experiments.run_all

# Regenerate the static dashboard's data from outputs/runs/*
dashboard:
	python -m finance_bench.visualizations.export_dashboard_data

# Full end-to-end pipeline:
# indexing -> gold alignment -> evaluation -> dashboard
full: qdrant-up index align evaluate dashboard