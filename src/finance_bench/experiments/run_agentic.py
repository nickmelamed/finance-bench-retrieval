import json

from dotenv import load_dotenv

from finance_bench.config.loaders import load_yaml_config
from finance_bench.retrieval.agentic import AgenticRetriever
from finance_bench.types.schemas import AgenticConfig, BM25Config, DenseConfig, DocumentChunk

load_dotenv()

config = load_yaml_config("retrieval/agentic.yaml", AgenticConfig)
bm25_config = load_yaml_config("retrieval/bm25.yaml", BM25Config)
dense_config = load_yaml_config("retrieval/dense.yaml", DenseConfig)

with open("data/processed/financebench_examples.json") as f:
    examples = json.load(f)

example = examples[0]

query = example["question"]

with open("data/processed/chunks.json") as f:
    raw_chunks = json.load(f)

chunks = [DocumentChunk.model_validate(chunk) for chunk in raw_chunks]

retriever = AgenticRetriever(
    chunks=chunks,
    config=config,
    bm25_config=bm25_config,
    dense_config=dense_config,
)

results = retriever.retrieve(query)

print(f"Question: {query}\n")

for result in results:
    print("=" * 80)
    print(result.score)
    print(result.text[:500])

prompt_tokens, completion_tokens = retriever.last_retrieval_usage()

print(f"\nRetrieval-time tokens: {prompt_tokens} prompt / {completion_tokens} completion")
