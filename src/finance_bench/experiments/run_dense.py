import os
import json

from dotenv import load_dotenv

from finance_bench.retrieval.dense import DenseRetriever
from finance_bench.types.schemas import DenseConfig
from finance_bench.config.loaders import load_yaml_config

config = load_yaml_config(
    "retrieval/dense.yaml",
    DenseConfig
)

with open(
    "data/processed/financebench_examples.json"
) as f:
    examples = json.load(f)


example = examples[0]

query = example["question"]

load_dotenv()

retriever = DenseRetriever(
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