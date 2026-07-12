import json
import os

from dotenv import load_dotenv

from finance_bench.retrieval.hybrid import HybridRetriever
from finance_bench.config.loaders import load_yaml_config
from finance_bench.types.schemas import HybridConfig

config = load_yaml_config(
    "retrieval/hybrid.yaml",
    HybridConfig
)
load_dotenv()

with open(
    "data/processed/financebench_examples.json"
) as f:
    examples = json.load(f)


example = examples[0]

query = example["question"]


with open("data/processed/chunks.json") as f:
    chunks = json.load(f)


retriever = HybridRetriever(
    chunks=chunks,
    embedding_model=os.getenv(
        "EMBEDDING_MODEL"
    ),
    top_k=config.top_k,
)


results = retriever.retrieve(query)


for result in results:
    print("=" * 80)
    print(result.score)
    print(result.text[:500])