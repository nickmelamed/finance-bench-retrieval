import json
import os

from dotenv import load_dotenv

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.agentic import AgenticGrepRetriever

load_dotenv()

with open("data/processed/chunks.json") as f:
    chunks = json.load(f)

# BM25 smoke test 

retriever = BM25Retriever(
    chunks=chunks,
    top_k=5,
)

results = retriever.retrieve(
    "What risks did management mention?"
)


# Dense Retrieval Smoke Test 

retriever = DenseRetriever(
    embedding_model=os.getenv(
        "EMBEDDING_MODEL"
    ),
    top_k=5,
)

results = retriever.retrieve(
    "What risks did management mention?"
)



# Hybrid Retrieval Smoke Test

retriever = HybridRetriever(
    chunks=chunks,
    embedding_model=os.getenv(
        "EMBEDDING_MODEL"
    ),
    top_k=5,
)

results = retriever.retrieve(
    "What risks did management mention?"
)

# Agentic Retrieval Smoke Test

retriever = AgenticGrepRetriever(
    chunks=chunks,
    top_k=5,
)

results = retriever.retrieve(
    "What risks did management mention?"
)

len(results)

# Retrieval consistency 

results_1 = retriever.retrieve(
    "revenue growth"
)

results_2 = retriever.retrieve(
    "revenue growth"
)

assert (
    results_1[0].chunk_id
    == results_2[0].chunk_id
)

# Top-k validation 

retriever = BM25Retriever(
    chunks=chunks,
    top_k=3,
)

results = retriever.retrieve(
    "operating margin"
)

assert len(results) == 3
