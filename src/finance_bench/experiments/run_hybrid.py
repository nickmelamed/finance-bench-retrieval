import json

from dotenv import load_dotenv

from finance_bench.retrieval.hybrid import HybridRetriever
from finance_bench.config.loaders import load_yaml_config
from finance_bench.types.schemas import BM25Config, DenseConfig, DocumentChunk, HybridConfig

load_dotenv()

hybrid_config = load_yaml_config(
    "retrieval/hybrid.yaml",
    HybridConfig
)

bm25_config = load_yaml_config(
    "retrieval/bm25.yaml",
    BM25Config
)

dense_config = load_yaml_config(
    "retrieval/dense.yaml",
    DenseConfig
)

with open(
    "data/processed/financebench_examples.json"
) as f:
    examples = json.load(f)


example = examples[0]

query = example["question"]


with open("data/processed/chunks.json") as f:
    raw_chunks = json.load(f)

chunks = [DocumentChunk.model_validate(chunk) for chunk in raw_chunks]


retriever = HybridRetriever(
    chunks=chunks,
    hybrid_config=hybrid_config,
    bm25_config=bm25_config,
    dense_config=dense_config,
)


results = retriever.retrieve(query)


for result in results:
    print("=" * 80)
    print(result.score)
    print(result.text[:500])
