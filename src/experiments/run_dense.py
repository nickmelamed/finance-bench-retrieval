import os
import json

from dotenv import load_dotenv

from src.retrieval.dense import DenseRetriever
from src.types.schemas import DenseConfig
from src.config.loaders import load_yaml_config

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