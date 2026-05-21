import json

from src.retrieval.agentic import AgenticGrepRetriever
from src.config.loaders import load_yaml_config
from src.types.schemas import AgenticConfig

config = load_yaml_config(
    "retrieval/agentic.yaml",
    AgenticConfig
)

with open(
    "data/processed/financebench_examples.json"
) as f:
    examples = json.load(f)


example = examples[0]

query = example["question"]


with open("data/processed/chunks.json") as f:
    chunks = json.load(f)


retriever = AgenticGrepRetriever(
    chunks=chunks,
    config=config
)


results = retriever.retrieve(query)


for result in results:
    print("=" * 80)
    print(result.score)
    print(result.text[:500])