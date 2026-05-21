# FinanceBench Retrieval Evaluation

A production-grade retrieval evaluation framework for comparing:

- BM25 retrieval
- Dense vector retrieval
- Hybrid retrieval
- Agentic grep-style retrieval

on the FinanceBench benchmark.

---

# Objective

Evaluate retrieval systems under token-efficiency constraints.

Primary metric:

Token Efficiency = Total Tokens / Correct Answers

---

# System Architecture

Question
→ Retriever
→ Retrieved Chunks
→ Claude Sonnet
→ Correctness Judge
→ Token Accounting
→ Visualization

---

# Features

## Retrieval

- Sparse retrieval
- Dense retrieval
- Reciprocal rank fusion
- Agentic retrieval
- Qdrant vector search

## Evaluation

- Claude-based correctness grading
- Bootstrap confidence intervals
- Token accounting
- Failure analysis
- Retrieval diagnostics
- Experiment tracking

## Visualization

- Accuracy CI plots
- Pareto frontier plots
- Failure mode distributions

---

# Analysis

Bootstrap confidence intervals are computed using:

- sampling with replacement
- 10,000 bootstrap iterations
- percentile confidence intervals

---

## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Environmental Variables 

Copy:

```bash
cp .env.example .env
```

Then populate:
* ANTHROPIC_API_KEY
* QDRANT_URL
* QDRANT_API_KEY

### Setup 

The Makefile contains the necessary commands, with 

```bash
make full
```

if you need to perform ingestion + indexing, or 

```bash 
make evaluate
``` 

if the corpus already exists. 

## Repo Structure 

* data → original dataset and chunks created 
* src/ingest → chunking + preprocessing
* src/retrieval → retrieval backends
* src/llm → Anthropic wrappers + token tracking
* src/evaluation → metrics + bootstrap statistics
* src/visualization → plots + tables
* outputs → experiment artifacts


## Reproducibility 

All experiments:
* use deterministic seeds
* cache LLM responses
* log token usage
* use fixed prompts
* use structured configs





