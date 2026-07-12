import json

from dotenv import load_dotenv

from finance_bench.config.loaders import load_yaml_config
from finance_bench.retrieval.bm25 import BM25Retriever
from finance_bench.types.schemas import BM25Config, DocumentChunk

load_dotenv()

config = load_yaml_config(
    "retrieval/bm25.yaml",
    BM25Config
)


QUERY = "What risks did management mention?"


with open("data/processed/chunks.json") as f:
    raw_chunks = json.load(f)

chunks = [DocumentChunk.model_validate(chunk) for chunk in raw_chunks]


retriever = BM25Retriever(
    chunks=chunks,
    config=config
)


results = retriever.retrieve(QUERY)


for result in results:
    print("=" * 80)
    print(result.score)
    print(result.text[:500])
