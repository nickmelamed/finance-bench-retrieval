import json

from src.retrieval.bm25 import BM25Retriever
from src.config.loaders import load_yaml_config
from src.types.schemas import BM25Config

config = load_yaml_config(
    "retrieval/bm25.yaml",
    BM25Config
)


QUERY = "What risks did management mention?"


with open("data/processed/chunks.json") as f:
    chunks = json.load(f)


retriever = BM25Retriever(
    chunks=chunks,
    config=config
)


results = retriever.retrieve(QUERY)


for result in results:
    print("=" * 80)
    print(result.score)
    print(result.text[:500])