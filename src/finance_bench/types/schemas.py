from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# core retrieval/data models 


class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str
    score: float
    retrieval_method: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QAResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    question: str

    gold_answer: str
    generated_answer: str

    correct: bool

    retrieval_method: str

    retrieved_chunks: List[str]

    latency_seconds: Optional[float] = None

    token_usage: Optional["TokenUsage"] = None


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    retrieval_tokens: int = 0
    reranking_tokens: int = 0

    total_tokens: int = 0


class ExperimentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str

    accuracy: float
    token_efficiency: float

    total_tokens: int
    correct_answers: int

    avg_latency_seconds: Optional[float] = None

    recall_at_k: Optional[float] = None
    mrr: Optional[float] = None
    ndcg: Optional[float] = None


# experiment yaml configs 


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    max_questions: int = 150


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_size: int = Field(
        default=350,
        gt=0,
    )

    chunk_overlap: int = Field(
        default=50,
        ge=0,
    )


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k: int = 5


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    temperature: float = 0.0
    max_tokens: int = 512


class BootstrapConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True

    n_bootstrap: int = 10000

    confidence_level: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
    )


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42

    dataset: DatasetConfig

    chunking: ChunkingConfig

    retrieval: RetrievalConfig

    llm: LLMConfig

    bootstrap: BootstrapConfig


# retrieval method configs 

class PromptTemplateConfig(BaseModel):
    template: str

class BaseRetrieverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str

    top_k: int = Field(
        default=5, 
        gt=0
    )


# BM25


class BM25Config(BaseRetrieverConfig):
    model_config = ConfigDict(extra="forbid")

    name: Literal["bm25"]

    k1: float = Field(default=1.5, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)


# Dense 


class DenseConfig(BaseRetrieverConfig):
    model_config = ConfigDict(extra="forbid")

    name: Literal["dense"]

    embedding_model: str

    similarity_metric: Literal[
        "cosine",
        "dot",
        "euclidean",
    ] = "cosine"

    normalize_embeddings: bool = True


# Hybrid 


class HybridConfig(BaseRetrieverConfig):
    model_config = ConfigDict(extra="forbid")

    name: Literal["hybrid"]

    fusion: Literal[
        "reciprocal_rank_fusion",
        "weighted_sum",
    ] = "reciprocal_rank_fusion"

    rrf_k: int = Field(default=60, gt=0)


# Agentic


class AgenticConfig(BaseRetrieverConfig):
    model_config = ConfigDict(extra="forbid")

    name: Literal["agentic"]

    # which Claude model drives the tool-use loop

    model: str = "claude-sonnet-4-6"

    # loop bounds

    max_tool_calls: int = Field(
        default=4,
        gt=0,
    )

    max_tokens_per_turn: int = Field(
        default=1024,
        gt=0,
    )

    # tool behavior

    search_top_k: int = Field(
        default=5,
        gt=0,
    )

    neighbor_window: int = Field(
        default=1,
        ge=0,
    )

    # chunk preview length inside tool results - the agent only needs
    # enough to judge relevance during exploration; the final QA answer
    # always uses the full untruncated chunk text regardless of this
    tool_result_preview_chars: int = Field(
        default=600,
        gt=0,
    )

    use_dense_tool: bool = True

    use_rerank_tool: bool = True

    # context management (prompt caching + trimming)

    keep_recent_tool_turns: int = Field(
        default=5,
        gt=0,
    )

    # diagnostics

    verbose: bool = False


# Legacy Generation Configs 


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    temperature: float
    max_tokens: int


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: str
    seed: int


class RepairConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    max_attempts: int


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelConfig
    sampling: SamplingConfig
    repair: RepairConfig